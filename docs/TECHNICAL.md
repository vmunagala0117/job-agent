# Technical Documentation — Job Agent

This document explains the internal code flow of the Job Agent system: how an HTTP request enters the system, how intent classification and routing work, how tool calls are dispatched, and how each module connects to the others.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Entry Points](#entry-points)
   - [server.py — Agent Server SDK](#serverpy--agent-server-sdk)
   - [webapp.py — FastAPI Web Application](#webappy--fastapi-web-application)
3. [Agent Construction: workflows.py](#agent-construction-workflowspy)
   - [Service Initialization](#service-initialization)
   - [WorkflowBuilder Wiring](#workflowbuilder-wiring)
   - [CoordinatorExecutor Internals](#coordinatorexecutor-internals)
4. [Request Lifecycle: @handler](#request-lifecycle-handler)
   - [Step 1 — Profile Preload](#step-1--profile-preload)
   - [Step 2 — Intent Classification with Logprobs](#step-2--intent-classification-with-logprobs)
   - [Step 3 — Specialist Delegation](#step-3--specialist-delegation)
   - [Step 4 — Audit Logging (OTel Spans)](#step-4--audit-logging-otel-spans)
   - [Step 5 — Token Usage Tracking](#step-5--token-usage-tracking)
   - [Step 6 — Final Output with Metadata](#step-6--final-output-with-metadata)
5. [Tool Execution: How Agent Framework Calls Your Code](#tool-execution-how-agent-framework-calls-your-code)
6. [Web Application: webapp.py](#web-application-webappy)
   - [Lifecycle & Agent Initialization](#lifecycle--agent-initialization)
   - [Chat API](#chat-api)
   - [Profile Management](#profile-management)
   - [Resume Upload](#resume-upload)
   - [Feedback System](#feedback-system)
   - [Trace Logging](#trace-logging)
7. [Observability & Enterprise Monitoring](#observability--enterprise-monitoring)
   - [OpenTelemetry Setup](#opentelemetry-setup)
   - [Structured Audit Spans](#structured-audit-spans)
   - [Classifier Confidence (Logprobs)](#classifier-confidence-logprobs)
   - [User Feedback Tracking](#user-feedback-tracking)
   - [KQL Queries for Application Insights](#kql-queries-for-application-insights)
8. [Module Reference](#module-reference)
9. [Data Models](#data-models)
10. [Key Design Decisions](#key-design-decisions)

---

## System Overview

```
HTTP POST /api/chat (ChatRequest)
  │
  ▼
FastAPI webapp.py
  │
  ├──  Manage session (conversation history)
  ├──  Build ChatMessage[] from history
  │
  ▼
WorkflowAgent.run(messages)
  │
  ▼
Workflow engine ──► CoordinatorExecutor.handle(messages, ctx)
  │
  ├──► _ensure_profile()                   Load profile from PostgreSQL
  │
  ├──► Direct OpenAI call (logprobs)       Classify intent → JOB_SEARCH or APP_PREP
  │    with confidence score               e.g. "JOB_SEARCH (confidence: 98.6%)"
  │
  ├──► self.job_search_agent.run(messages)  Full agentic tool-calling loop (11 tools)
  │      or
  ├──► self.app_prep_agent.run(messages)    Full agentic tool-calling loop (3 tools)
  │
  ├──► Audit: emit OTel spans              tool.name, tool.arguments, tool.result_preview
  ├──► Track token usage                   input + output tokens
  └──► ctx.yield_output(text + metadata)   JSON metadata suffix for the webapp
```

The system has three LLM-backed agents running inside a **single** `Executor` node. The Agent Framework sees one node, but internally the `CoordinatorExecutor` manages all three sub-agents programmatically.

---

## Entry Points

### server.py — Agent Server SDK

The original headless server using Azure Agent Server SDK's `from_agent_framework()`:

```python
def main():
    agent = asyncio.run(_create_agent())
    from_agent_framework(agent).run()
```

Accepts `POST` requests with `ChatMessage[]` JSON, returns the agent's text response. No UI — designed for programmatic integration.

### webapp.py — FastAPI Web Application

The primary entry point, providing a full web UI with chat interface:

```python
# Run with: python -m job_agent.webapp
# or:       uvicorn job_agent.webapp:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve the HTML/JS/CSS chat UI |
| `GET` | `/health` | Health check + session info |
| `POST` | `/api/chat` | Send message, get agent response |
| `POST` | `/api/upload-resume` | Upload PDF/DOCX resume |
| `GET` | `/api/traces` | Get trace log entries for session |
| `GET` | `/api/profiles` | List all user profiles |
| `GET` | `/api/profiles/{id}` | Get profile details |
| `POST` | `/api/profiles/save` | Create/update profile |
| `POST` | `/api/profiles/select` | Switch active profile |
| `POST` | `/api/feedback` | Submit thumbs up/down on response |
| `GET` | `/api/feedback` | List feedback entries |
| `POST` | `/api/chat/reset` | Reset conversation history |

---

## Agent Construction: workflows.py

### Service Initialization

`create_agent()` calls `_init_services()` which creates the 5 shared service singletons:

```
_init_services(use_database=True)
  │
  ├── JobStore                 PostgresJobStore (asyncpg + pgvector)
  │                            or InMemoryJobStore (fallback)
  │
  ├── JobIngestionProvider     SerpAPIProvider (real Google Jobs data)
  │                            or MockJobProvider (test data)
  │
  ├── RankingService           Uses AzureOpenAIEmbeddingService (text-embedding-3-small)
  │                            or MockEmbeddingService
  │
  ├── NotificationService      Sends to console, email, Teams, or Slack
  │
  └── ApplicationPrepService   LLM-based resume/cover letter generation
```

Each service is a plain Python class instantiated once. They're passed into the `CoordinatorExecutor` constructor where they become available to tool methods via dependency injection (instance attributes on `JobSearchTools` and `AppPrepTools`).

### WorkflowBuilder Wiring

```python
def build_agent(client, store, provider, ranking_service, ...):
    workflow = (
        WorkflowBuilder()
        .register_executor(
            lambda: CoordinatorExecutor(client, store, provider, ...),
            name="coordinator",
        )
        .set_start_executor("coordinator")
        .build()
    )
    return workflow.as_agent()
```

This is the Agent Framework's DAG-based workflow engine. Even though we only have **one node**, the pattern matters because:

1. **`register_executor(factory, name)`** — Registers a **factory lambda**, not an instance. The `CoordinatorExecutor` constructor doesn't run until `.build()` resolves the executor graph.
2. **`.set_start_executor("coordinator")`** — Tells the engine which node handles incoming messages.
3. **`.build()`** — Returns a `Workflow` object. Calls the lambda, instantiates `CoordinatorExecutor`, wires it into the execution graph.
4. **`.as_agent()`** — Wraps the `Workflow` into a `WorkflowAgent` that implements `AgentProtocol`.

### CoordinatorExecutor Internals

The constructor creates **three `ChatAgent` sub-agents** from the same `AzureOpenAIChatClient`, plus a **direct `AsyncAzureOpenAI` client** for logprobs-based classification:

```python
class CoordinatorExecutor(Executor):
    def __init__(self, client, store, provider, ranking_service, ...,
                 openai_client=None, deployment_name=""):
        # Tool containers (dependency injection)
        job_tools = JobSearchTools(store, provider, ranking_service, notification_service)
        app_tools = AppPrepTools(store, application_prep_service)

        # Direct OpenAI client for logprobs classifier
        self._openai_client = openai_client
        self._deployment = deployment_name

        # Classifier agent (no tools, cheap routing call)
        self.classifier = client.create_agent(name="coordinator", instructions=CLASSIFIER_INSTRUCTIONS)

        # Job Search Agent — 11 tools
        self.job_search_agent = client.create_agent(
            name="job_search_agent", instructions=JOB_SEARCH_INSTRUCTIONS,
            tools=[job_tools.search_jobs, job_tools.rank_saved_jobs, ...],
        )

        # App Prep Agent — 3 tools
        self.app_prep_agent = client.create_agent(
            name="application_prep_agent", instructions=APP_PREP_INSTRUCTIONS,
            tools=[app_tools.prepare_application, app_tools.analyze_job_fit, ...],
        )
```

The direct `AsyncAzureOpenAI` client bypasses the framework's abstraction specifically to access `logprobs` and `top_logprobs` parameters, which the Agent Framework's `ChatAgent` doesn't expose.

---

## Request Lifecycle: @handler

### Step 1 — Profile Preload

```python
await self._ensure_profile()
```

On the **first request only** (guarded by `self._initialized` flag), loads the user's saved profile from PostgreSQL:

```
store.get_default_profile()  →  SELECT * FROM profiles ORDER BY updated_at DESC LIMIT 1
```

If found, calls `set_current_profile(profile)` which writes to a module-level global in `tools.py`.

### Step 2 — Intent Classification with Logprobs

When a direct OpenAI client is available, classification uses `logprobs` for confidence scoring:

```python
resp = await self._openai_client.chat.completions.create(
    model=self._deployment,
    messages=classify_messages,
    logprobs=True,
    top_logprobs=3,
    max_completion_tokens=150,
)
# Extract confidence: exp(logprob) * 100
confidence = round(math.exp(first_token.logprob) * 100, 1)
```

The classifier prompt instructs the model to output only `JOB_SEARCH` or `APP_PREP`. With `top_logprobs=3`, we capture the probability distribution across alternatives.

**OTel span attributes emitted:**
- `classifier.intent` — The chosen intent
- `classifier.confidence_pct` — Probability as percentage (e.g., 98.6%)
- `classifier.alternatives` — Other top tokens and their probabilities
- `classifier.user_message` — First 200 chars of the user's message

Falls back to the Agent Framework's `self.classifier.run(messages)` if the direct client is unavailable.

### Step 3 — Specialist Delegation

```python
if "APP_PREP" in intent:
    response = await self.app_prep_agent.run(messages)
else:
    response = await self.job_search_agent.run(messages)
```

`ChatAgent.run(messages)` triggers the Agent Framework's **full agentic tool-calling loop** (managed by the `@use_function_invocation` decorator on `AzureOpenAIChatClient`):

```
1. Send messages + tool schemas to Azure OpenAI
2. Model responds with text and/or tool_calls
3. If tool_calls:
   a. Deserialize arguments from JSON
   b. Call the Python method (your tool code)
   c. Append FunctionResultContent to conversation
   d. Go back to step 1 (up to 40 iterations)
4. If text only → return AgentRunResponse
```

The framework handles this transparently. Your tool methods receive deserialized Python arguments and return strings.

### Step 4 — Audit Logging (OTel Spans)

After the specialist responds, the handler iterates through `response.messages` to extract tool call records:

```python
for message in response.messages:
    for c in message.contents:
        if getattr(c, "type", None) == "function_call":       # FunctionCallContent
            tool_name = c.name
            args_str = str(c.arguments)[:500]
            # Match with FunctionResultContent by call_id
            ...
            # Emit structured OTel span
            with _tracer.start_as_current_span("tool_call") as tool_span:
                tool_span.set_attribute("tool.name", tool_name)
                tool_span.set_attribute("tool.arguments", args_str)
                tool_span.set_attribute("tool.result_preview", tool_result[:300])
                tool_span.set_attribute("tool.agent", agent_name)
                tool_span.set_attribute("audit.type", "tool_invocation")
```

**Important**: Agent Framework uses `FunctionCallContent` (with `name`, `call_id`, `arguments`) and `FunctionResultContent` (with `call_id`, `result`) — these are framework-specific types, not OpenAI's raw format. The `.type` attribute (`"function_call"` / `"function_result"`) is the most reliable discriminator for detecting tool usage in `response.messages`.

### Step 5 — Token Usage Tracking

Token usage is aggregated from two sources:
1. **Classifier call** — from the direct OpenAI response's `usage` object
2. **Specialist agent** — from `response.usage_details` (Agent Framework's tracked usage)

```python
usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
# ... add classifier tokens ...
# ... add specialist tokens ...
```

### Step 6 — Final Output with Metadata

The response text is augmented with a JSON metadata suffix that the webapp extracts and strips:

```python
metadata = {
    "usage": usage_totals,
    "elapsed_ms": elapsed_ms,
    "classifier_confidence": confidence,
    "agent": agent_name,
    "tool_count": len(tool_calls),
}
output = f"{response_text}\n<!--METADATA:{json.dumps(metadata)}-->"
await ctx.yield_output(output)
```

The webapp's chat endpoint parses `<!--METADATA:...-->` from the response, strips it from the displayed text, and includes the parsed fields in `ChatResponse`.

---

## Tool Execution: How Agent Framework Calls Your Code

When you pass `tools=[job_tools.search_jobs]` to `client.create_agent()`, the framework:

1. **Introspects the method signature:**
   ```python
   async def search_jobs(self, query: str, location: Optional[str] = None, ...) -> str:
   ```

2. **Generates OpenAI tool JSON schema** from type hints and docstrings

3. **At runtime**, when the model decides to call a tool:
   - Model outputs: `{"tool_calls": [{"function": {"name": "search_jobs", "arguments": "..."}}]}`
   - Framework deserializes and calls `await job_tools.search_jobs(query="AI", location="Seattle")`
   - The `self` binding gives each tool access to shared services (`self.store`, `self.provider`, etc.)
   - Framework captures the return value and creates a `FunctionResultContent` message
   - Framework sends the updated conversation back to the model for the next turn

**The `self` binding is key**: `job_tools.search_jobs` is a **bound method** — it carries a reference to the `JobSearchTools` instance, which holds `self.store`, `self.provider`, `self.ranking_service`, and `self.notification_service`. This is how tools access shared services without global state.

---

## Web Application: webapp.py

### Lifecycle & Agent Initialization

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    agent, store, ranking_service = await create_agent(client, use_database=True)
    app.state.agent = agent
    app.state.store = store
    ...
```

The FastAPI `lifespan` handler initializes the agent once at startup. The agent, store, and ranking service are stored on `app.state` and reused across all requests.

### Chat API

`POST /api/chat` receives `ChatRequest` (message + session_id), maintains conversation history per session, runs the agent, and returns `ChatResponse` including:
- `response` — The agent's text (metadata suffix stripped)
- `session_id` — For conversation continuity
- `usage` — Token counts (input, output, total)
- `classifier_confidence` — Routing confidence percentage
- `elapsed_ms` — End-to-end latency
- `agent` — Which specialist handled the request
- `tool_count` — Number of tool calls made

### Profile Management

Profiles support create, read, update, select, and list operations:
- **Save**: `POST /api/profiles/save` — Creates new or updates existing. Resume-extracted fields (current_title, summary, skills, years_experience) are preserved on update.
- **Select**: `POST /api/profiles/select` — Switches the active profile for the session.
- **UI**: Modal form with Name, Email, Resume upload (top), read-only "Extracted from resume" summary, optional preferences (desired titles, locations, remote preference, salary, industries).

### Resume Upload

`POST /api/upload-resume` — Accepts PDF/DOCX files, extracts text via `ResumeParser`, auto-extracts skills/experience/title/summary, creates embeddings, stores full profile in PostgreSQL (no text truncation).

### Feedback System

Thumbs up/down on agent responses:
- `POST /api/feedback` — Records `ResponseFeedback` with rating, message hash, optional comment
- `GET /api/feedback` — Lists all entries
- OTel span emitted with `feedback.rating` and `feedback.response_preview` attributes

### Trace Logging

`GET /api/traces` returns structured trace entries captured by `SessionTraceHandler` — a custom `logging.Handler` that captures `[TRACE]`-prefixed log messages per session. The UI renders these as a collapsible trace panel.

---

## Observability & Enterprise Monitoring

### OpenTelemetry Setup

The webapp configures OpenTelemetry with Azure Monitor export:

```python
from azure.monitor.opentelemetry import configure_azure_monitor

configure_azure_monitor(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
    enable_live_metrics=True,
)
```

All spans from `_tracer = trace.get_tracer("job_agent.workflows")` are automatically exported to Application Insights.

### Structured Audit Spans

Every tool invocation during an agent run emits a `tool_call` span with:
- `tool.name` — Function name (e.g., `search_jobs`, `get_profile`)
- `tool.arguments` — Serialized arguments (truncated to 500 chars)
- `tool.result_preview` — First 300 chars of the tool's return value
- `tool.agent` — Which specialist agent made the call
- `audit.type` — Always `"tool_invocation"`

The classifier emits a `classifier` span with:
- `classifier.intent` — JOB_SEARCH or APP_PREP
- `classifier.confidence_pct` — Probability percentage
- `classifier.alternatives` — Top-3 logprob alternatives
- `classifier.user_message` — Truncated user input

### Classifier Confidence (Logprobs)

The first token's log-probability is converted to a percentage: `round(exp(logprob) * 100, 1)`. This enables monitoring for low-confidence classifications that might indicate ambiguous or misrouted requests.

### User Feedback Tracking

Feedback is tracked as OTel spans with `feedback.rating`, `feedback.response_preview`, and `feedback.session_id` attributes.

### KQL Queries for Application Insights

See `docs/kql-queries.kql` for 14 ready-to-run KQL queries covering:
- All spans overview, classifier decisions/confidence, low-confidence flags (< 80%)
- Tool audit trail, usage frequency, agent routing distribution
- Response times (P50/P95/P99), feedback summary, thumbs-down investigation
- End-to-end timeline, error rates, slow tools (> 5s), hourly traffic, daily sessions

---

## Evaluation Framework

The `evals/` directory provides automated quality assessment using the
**Azure AI Evaluation SDK** and custom code-based evaluators.

### Architecture

```
evals/
├── golden_dataset.jsonl           # 30 labeled queries (expected agent + tools)
├── response_quality_dataset.jsonl  # Agent responses with tool result context
├── eval_classifier.py             # Classification accuracy evaluation
├── eval_response_quality.py       # LLM-as-judge (unified evaluate() API)
├── evaluators.py                  # Custom evaluators (code-based)
└── results/                       # Output reports (JSON, per-row data)
```

### LLM-as-Judge Evaluators (Built-in)

Uses the Azure AI Evaluation SDK's `evaluate()` API with four built-in evaluators:

| Evaluator | What It Measures | Required Data |
|-----------|-----------------|---------------|
| `RelevanceEvaluator` | Does the response address the user's question? | query, response |
| `GroundednessEvaluator` | Are claims backed by tool results (not hallucinated)? | response, context |
| `CoherenceEvaluator` | Is the response well-structured and logical? | query, response |
| `FluencyEvaluator` | Is the language grammatically correct? | response |

These evaluators use the same Azure OpenAI deployment as the agent. Results
include per-row scores (1–5) and aggregate means.

### Custom Code-Based Evaluators

`evals/evaluators.py` defines three evaluators compatible with the `evaluate()` API:

| Evaluator | Score | Description |
|-----------|-------|-------------|
| `ResponseLengthEvaluator` | 0.0 / 0.5 / 1.0 | Flags responses < 50 chars (failure) or < 100 chars (marginal) |
| `ToolUsageEvaluator` | 0.0 – 1.0 | Recall: what fraction of expected tools were actually invoked |
| `ClassificationEvaluator` | 0.0 / 1.0 | Binary: did the classifier route to the correct agent? |

### Classifier Accuracy Test Suite

`evals/eval_classifier.py` runs the production logprobs classifier against 30 labeled
queries from `golden_dataset.jsonl`:

1. Sends each query through the exact same classifier prompt + model
2. Extracts the predicted intent and logprobs confidence
3. Compares against the expected agent label
4. Produces a confusion matrix, misclassification report, and JSON results
5. Exits non-zero if accuracy < 90% (CI gate)

### Feedback Analytics

`GET /api/feedback/analytics` aggregates thumbs up/down data:

```json
{
  "total": 42,
  "thumbs_up": 38,
  "thumbs_down": 4,
  "satisfaction_rate": 90.5,
  "unique_sessions": 12,
  "sessions_with_negative": 3,
  "recent_negative": [...],
  "timeline": [{"date": "2026-02-11", "total": 5, "up": 4, "down": 1}]
}
```

### Running Evaluations

```bash
# Classification accuracy (calls Azure OpenAI, ~30 queries)
python evals/eval_classifier.py --verbose

# Response quality (LLM-as-judge, ~6 responses)
python evals/eval_response_quality.py

# Custom dataset
python evals/eval_response_quality.py --dataset evals/my_data.jsonl
```

---

## Module Reference

### config.py

| Class | Key env vars | Purpose |
|-------|-------------|---------|
| `AzureOpenAIConfig` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_KEY` | Chat model + embeddings |
| `SerpAPIConfig` | `SERPAPI_API_KEY` | Real job data provider |
| `DatabaseConfig` | `DATABASE_URL` or `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` | PostgreSQL connection |
| `AppConfig` | Composes all above | Top-level config, `.load()` factory |

### clients.py

`build_azure_openai_client(config) → AzureOpenAIChatClient` — Chooses auth: API key or `DefaultAzureCredential`.

### models.py

| Model | Key fields | Used by |
|-------|-----------|---------|
| `Job` | title, company, location, description, skills, embedding, status | Store, ranking, tools |
| `UserProfile` | name, email, skills, resume_text, embedding, desired_titles, preferred_locations, remote_preference, min_salary, industries | Store, ranking, tools, profile UI |
| `JobSearchCriteria` | query, location, remote_only, min_salary, date_posted | Provider, store |
| `RankedJob` | job, score, justification, score breakdown | Ranking, notifications |
| `JobFeedback` | job_id, feedback_type, notes | Store, tools |
| `ApplicationPackage` | job, profile, resume_suggestions, cover_letter, intro_email, recruiters | Store, app prep |
| `ResponseFeedback` | rating (UP/DOWN), message_hash, comment | Feedback tracking |

Enums: `JobStatus`, `DatePosted`, `FeedbackType`, `NotificationChannel`, `ResponseRating`

### store.py

Abstract `JobStore` with `InMemoryJobStore` (testing) and `PostgresJobStore` (asyncpg + pgvector).

Key methods: `add_many`, `list_all`, `update_status`, `save_profile`, `get_default_profile`, `get_profile`, `list_profiles`, `update_job_embeddings`, `save_feedback`, `save_application_package`.

### providers.py

`SerpAPIProvider` (Google Jobs via httpx) and `MockJobProvider` (hardcoded test data).

### ranking.py

`AzureOpenAIEmbeddingService` (text-embedding-3-small, 1536d) + `RankingService` with composite scoring: cosine similarity (50%) + skill overlap (25%) + location match (15%) + salary match (10%).

### resume_parser.py

PDF (pymupdf), DOCX (python-docx), TXT. Regex-based extraction of name, email, skills, experience years, title. Full text stored without truncation.

### notifications.py

Console (always), Email (SMTP), Teams (Adaptive Cards webhook), Slack (Block Kit webhook).

### application_prep.py

`ApplicationPrepService` — Resume diff suggestions, cover letters (< 250 words), intro emails, and recruiter search (Proxycurl API if configured).

### tools.py

Module-level global for current user profile. Session-scoped for single-user operation.

### workflows.py

Core orchestration: instruction constants, `JobSearchTools` (11 tools), `AppPrepTools` (3 tools), `CoordinatorExecutor` with `@handler`, `create_agent()` factory.

---

## Data Models

### Embedding Pipeline

```
Profile creation → ResumeParser extracts text + skills
  → embed(resume_text + skills) → profile.embedding = float[1536]
  → store in PostgreSQL

Job search → SerpAPI fetch → store jobs
  → embed_batch(descriptions) → job.embedding = float[1536]
  → store embeddings in pgvector

Ranking → cosine_similarity(profile, job) + heuristic boosts
  → sorted RankedJob[] with justifications
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Classifier-based routing** (not `HandoffBuilder`) | `HandoffBuilder` is incompatible with `from_agent_framework()` HTTP serving — autonomous mode hits 100-superstep limit, human-in-loop returns empty `response.text` |
| **Direct `AsyncAzureOpenAI` for classification** | Agent Framework's `ChatAgent` doesn't expose `logprobs`/`top_logprobs` — needed for confidence scoring |
| **Bound methods as tools** | Gives tools access to shared services via `self.store`, `self.provider` without global state |
| **Module-level profile global** | Simplest approach for single-user; would use context-scoped storage for multi-tenant |
| **Single-node `WorkflowBuilder`** | Proper `Executor` lifecycle, `AgentProtocol` compliance, future extensibility |
| **`max_completion_tokens`** (not `max_tokens`) | Azure OpenAI gpt-5.2 requires the newer parameter name |
| **`FunctionCallContent.type == "function_call"` for audit** | Most reliable discriminator; framework types use `name`/`call_id`/`arguments`, not OpenAI's raw format |
| **Full resume text** (no truncation) | Resume text stored and retrieved in full from PostgreSQL; truncation was causing incomplete agent responses |
