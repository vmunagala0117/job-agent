# Technical Documentation — Job Agent

This document explains the internal code flow of the Job Agent system: how an HTTP request enters the system, how intent classification and routing work, how tool calls are dispatched, and how each module connects to the others.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Entry Point: server.py](#entry-point-serverpy)
3. [Agent Construction: workflows.py](#agent-construction-workflowspy)
   - [Service Initialization](#service-initialization)
   - [WorkflowBuilder Wiring](#workflowbuilder-wiring)
   - [CoordinatorExecutor Internals](#coordinatorexecutor-internals)
4. [Request Lifecycle: @handler](#request-lifecycle-handler)
   - [Step 1 — Profile Preload](#step-1--profile-preload)
   - [Step 2 — Intent Classification](#step-2--intent-classification)
   - [Step 3 — Specialist Delegation](#step-3--specialist-delegation)
   - [Step 4 — Response Streaming](#step-4--response-streaming)
   - [Step 5 — Final Output](#step-5--final-output)
5. [Tool Execution: How Agent Framework Calls Your Code](#tool-execution-how-agent-framework-calls-your-code)
6. [Module Reference](#module-reference)
   - [config.py](#configpy)
   - [clients.py](#clientspy)
   - [models.py](#modelspy)
   - [store.py](#storepy)
   - [providers.py](#providerspy)
   - [ranking.py](#rankingpy)
   - [resume_parser.py](#resume_parserpy)
   - [notifications.py](#notificationspy)
   - [application_prep.py](#application_preppy)
   - [tools.py](#toolspy)
7. [Data Models](#data-models)
8. [Key Design Decisions](#key-design-decisions)

---

## System Overview

```
HTTP POST (ChatMessage[])
  │
  ▼
from_agent_framework(agent).run()          Azure Agent Server SDK
  │
  ▼
WorkflowAgent.run(messages)                agent_framework
  │
  ▼
Workflow engine ──► CoordinatorExecutor.handle(messages, ctx)
  │
  ├──► self.classifier.run(messages)       Lightweight LLM call → "JOB_SEARCH" or "APP_PREP"
  │
  ├──► self.job_search_agent.run(messages) Full agentic tool-calling loop (11 tools)
  │      or
  ├──► self.app_prep_agent.run(messages)   Full agentic tool-calling loop (3 tools)
  │
  ├──► ctx.add_event(...)                  Streaming partial updates to the HTTP client
  └──► ctx.yield_output(response.text)     Final HTTP response body
```

The system has three LLM-backed agents running inside a **single** `Executor` node. The Agent Framework sees one node, but internally the `CoordinatorExecutor` manages all three sub-agents programmatically.

---

## Entry Point: server.py

```python
# server.py — full file, simplified
def main():
    agent = asyncio.run(_create_agent())   # 1. build the agent
    from_agent_framework(agent).run()       # 2. start HTTP server

async def _create_agent():
    config = AppConfig.load()               # reads .env
    client = build_azure_openai_client(config.azure_openai)
    return await create_agent(client, use_database=True)
```

**What happens at each step:**

| Step | Function | What it does |
|------|----------|-------------|
| `AppConfig.load()` | `config.py` | Reads `.env` file, builds `AzureOpenAIConfig`, `SerpAPIConfig`, `DatabaseConfig` dataclasses |
| `build_azure_openai_client()` | `clients.py` | Creates `AzureOpenAIChatClient` with either API key or `DefaultAzureCredential` (managed identity) |
| `create_agent()` | `workflows.py` | Async factory — initializes DB pool, services, builds the workflow (see below) |
| `from_agent_framework(agent)` | Azure Agent Server SDK | Wraps any `AgentProtocol`-compatible object as an HTTP server. Accepts `POST` requests with `ChatMessage[]`, returns the agent's text response |

`from_agent_framework()` is the bridge between the Agent Framework's in-process agent API and the network. It handles JSON serialization/deserialization of `ChatMessage` objects, streaming events as server-sent events, and error handling. Your code never touches HTTP directly.

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

1. **`register_executor(factory, name)`** — Registers a **factory lambda**, not an instance. The `CoordinatorExecutor` constructor doesn't run until `.build()` resolves the executor graph. This is important because `Executor.__init__()` has framework-internal setup that must happen during graph resolution.

2. **`.set_start_executor("coordinator")`** — Tells the engine which node handles incoming messages. In a multi-node graph you'd have edges between nodes; here we have just one.

3. **`.build()`** — Returns a `Workflow` object. Calls the lambda, instantiates `CoordinatorExecutor`, wires it into the execution graph.

4. **`.as_agent()`** — Wraps the `Workflow` into a `WorkflowAgent` that implements `AgentProtocol` (i.e., has `.run(messages)` and `.run_stream(messages)` methods). This is what `from_agent_framework()` needs.

### CoordinatorExecutor Internals

The constructor creates **three `ChatAgent` sub-agents** from the same `AzureOpenAIChatClient`:

```python
class CoordinatorExecutor(Executor):
    def __init__(self, client, store, provider, ranking_service, ...):
        # 1. Build tool containers (dependency injection)
        job_tools = JobSearchTools(store, provider, ranking_service, notification_service)
        app_tools = AppPrepTools(store, application_prep_service)

        # 2. Classifier agent — no tools, just text classification
        self.classifier = client.create_agent(
            name="coordinator",
            instructions=CLASSIFIER_INSTRUCTIONS,
        )

        # 3. Job Search Agent — 11 tools
        self.job_search_agent = client.create_agent(
            name="job_search_agent",
            instructions=JOB_SEARCH_INSTRUCTIONS,
            tools=[job_tools.search_jobs, job_tools.rank_saved_jobs, ...],
        )

        # 4. App Prep Agent — 3 tools
        self.app_prep_agent = client.create_agent(
            name="application_prep_agent",
            instructions=APP_PREP_INSTRUCTIONS,
            tools=[app_tools.prepare_application, app_tools.analyze_job_fit, ...],
        )

        super().__init__(id=id)  # Executor base requires an id
```

**How `client.create_agent()` handles the `tools=` parameter:**

The Agent Framework accepts plain Python async methods as tools. For each method, it:
1. Reads the **function name** → becomes the tool name
2. Reads the **docstring** → becomes the tool description
3. Reads the **type hints** on parameters → generates the JSON Schema for the tool's `parameters` property
4. Reads `Args:` in the docstring → becomes parameter descriptions in the schema
5. The **return type** (always `str`) → tool output is passed back to the model as a tool message

This is why every tool method has detailed type annotations and Google-style docstrings — they're not just for humans, they're the tool schema definition.

---

## Request Lifecycle: @handler

When the HTTP server receives a request, the call chain is:

```
HTTP POST → from_agent_framework → WorkflowAgent.run(messages)
  → Workflow._run_workflow_with_tracing()
    → _execute_with_message_or_checkpoint()
      → CoordinatorExecutor.execute(messages, context)
        → finds @handler-decorated method
          → handle(messages, ctx)
```

The `@handler` decorator marks which method the `Executor` base class should invoke. Only one handler per executor.

### Step 1 — Profile Preload

```python
await self._ensure_profile()
```

On the **first request only** (guarded by `self._initialized` flag), loads the user's saved profile from PostgreSQL:

```
store.get_default_profile()  →  SELECT * FROM profiles ORDER BY updated_at DESC LIMIT 1
```

If found, calls `set_current_profile(profile)` which writes to a **module-level global** in `tools.py`. This global is read by tool methods like `rank_saved_jobs` and `provide_feedback` that need the profile without requiring it as a parameter. (The global is process-scoped — fine for single-user, would need scoping for multi-tenant.)

### Step 2 — Intent Classification

```python
classification = await self.classifier.run(messages)
intent = (classification.text or "").strip().upper()
```

The classifier agent has **no tools** and a system prompt that says:

> Reply with ONLY the category name: JOB_SEARCH or APP_PREP. Nothing else.

This is a cheap, fast LLM call — roughly 100 output tokens. The model receives the full conversation history but only produces a single word. The response is parsed into a string and checked with `"APP_PREP" in intent`.

The default route is `JOB_SEARCH` (the `else` branch), so ambiguous requests like greetings or small talk go to the job search agent, which has more general capabilities.

### Step 3 — Specialist Delegation

```python
if "APP_PREP" in intent:
    response = await self.app_prep_agent.run(messages)
else:
    response = await self.job_search_agent.run(messages)
```

`ChatAgent.run(messages)` is the Agent Framework's **full agentic loop**:

```
┌─────────────────────────────────────────────────────┐
│  ChatAgent.run(messages)                            │
│                                                     │
│  1. Send messages + tool schemas to Azure OpenAI    │
│  2. Model responds with text and/or tool_calls      │
│  3. If tool_calls:                                  │
│     a. Deserialize arguments from JSON              │
│     b. Call the Python method (your tool code)      │
│     c. Append tool result as a tool message         │
│     d. Go back to step 1 (up to max iterations)    │
│  4. If text only → return AgentRunResponse          │
└─────────────────────────────────────────────────────┘
```

The model may call **multiple tools** in a single turn (parallel tool calling), and it may do **multiple rounds** (e.g., search → rank → notify). The framework handles all of this automatically. Your tool methods just receive deserialized Python arguments and return strings.

**Example flow for "Search for AI jobs in Seattle":**
```
User message → model sees 11 tool schemas → decides to call search_jobs
  → framework calls JobSearchTools.search_jobs(query="AI", location="Seattle, WA")
    → SerpAPIProvider.fetch_jobs(criteria)   [HTTP to Google Jobs]
    → store.add_many(jobs)                   [PostgreSQL INSERT]
    → ranking_service.embed_jobs(jobs)       [Azure OpenAI embeddings API]
    → store.update_job_embeddings(...)       [PostgreSQL UPDATE with pgvector]
    → returns "Found 10 jobs matching 'AI in Seattle': ..."
  → framework sends tool result back to model
  → model generates final text response with formatted job listings
```

### Step 4 — Response Streaming

```python
for message in response.messages:
    if message.role == Role.ASSISTANT and message.contents:
        text_parts = [
            c for c in message.contents
            if isinstance(c, TextContent) and c.text
        ]
        if text_parts:
            await ctx.add_event(AgentRunUpdateEvent(...))
```

The `response.messages` list contains **every message** from the agentic loop — including internal `FunctionCallContent` (the model requesting a tool call) and `FunctionResultContent` (the tool output). We only want `TextContent` from `ASSISTANT` messages.

`ctx.add_event()` pushes partial updates to the HTTP client via server-sent events, enabling **streaming** UX. Each event wraps the text in `AgentRunResponseUpdate` with a unique `response_id`.

The `isinstance(c, TextContent)` filter is critical: without it, you'd crash on `FunctionCallContent` objects which don't have a `.text` attribute.

### Step 5 — Final Output

```python
await ctx.yield_output(response.text)
```

`response.text` is a convenience property on `AgentRunResponse` that extracts the text from the **last assistant message**. `ctx.yield_output()` tells the workflow engine "this executor is done, here's the result" — this becomes the HTTP response body.

---

## Tool Execution: How Agent Framework Calls Your Code

When you pass `tools=[job_tools.search_jobs]` to `client.create_agent()`, the framework:

1. **Introspects the method signature:**
   ```python
   async def search_jobs(
       self,
       query: str,                           # required string
       location: Optional[str] = None,       # optional string
       remote_only: bool = False,            # optional boolean, default false
       min_salary: Optional[int] = None,     # optional integer
       max_results: int = 10,                # optional integer, default 10
       date_posted: Optional[str] = None,    # optional string
   ) -> str:
   ```

2. **Generates OpenAI tool JSON schema:**
   ```json
   {
     "type": "function",
     "function": {
       "name": "search_jobs",
       "description": "Search for new job listings matching the criteria.",
       "parameters": {
         "type": "object",
         "properties": {
           "query": {"type": "string", "description": "Job search query..."},
           "location": {"type": "string", "description": "Geographic location..."},
           "remote_only": {"type": "boolean", "description": "Set to True for remote..."},
           ...
         },
         "required": ["query"]
       }
     }
   }
   ```

3. **At runtime**, when the model decides to call a tool:
   - Model outputs: `{"tool_calls": [{"function": {"name": "search_jobs", "arguments": "{\"query\": \"AI\", \"location\": \"Seattle\"}"}}]}`
   - Framework parses the JSON arguments
   - Framework calls `await job_tools.search_jobs(query="AI", location="Seattle")`
   - The `self` parameter is the bound `JobSearchTools` instance (since we passed a bound method)
   - Framework captures the return value (a string)
   - Framework creates a tool message with the result and appends it to the conversation
   - Framework sends the updated conversation back to the model for the next turn

**The `self` binding is key**: `job_tools.search_jobs` is a **bound method** — it carries a reference to the `JobSearchTools` instance, which holds `self.store`, `self.provider`, `self.ranking_service`, and `self.notification_service`. This is how tools access shared services without global state.

---

## Module Reference

### config.py

Configuration dataclasses loaded from environment variables via `python-dotenv`:

| Class | env vars | Purpose |
|-------|----------|---------|
| `AzureOpenAIConfig` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_KEY` | Chat model + embeddings config |
| `SerpAPIConfig` | `SERPAPI_API_KEY` | Real job data provider |
| `DatabaseConfig` | `DATABASE_URL` or `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` | PostgreSQL connection |
| `AppConfig` | Composes all above | Top-level config, `.load()` factory |

### clients.py

Single function: `build_azure_openai_client(config) → AzureOpenAIChatClient`

Chooses auth strategy: if `api_key` is set, uses `AzureKeyCredential`; otherwise uses `DefaultAzureCredential` (supports `az login`, managed identity, etc.).

### models.py

Pure dataclasses, no business logic. All use `@dataclass` with `field(default_factory=...)` for mutable defaults.

| Model | Key fields | Used by |
|-------|-----------|---------|
| `Job` | title, company, location, description, skills, embedding, status | Store, ranking, tools |
| `UserProfile` | name, skills, resume_text, embedding, preferences | Store, ranking, tools |
| `JobSearchCriteria` | query, location, remote_only, min_salary, date_posted | Provider, store |
| `RankedJob` | job, score, justification, score breakdown | Ranking, notifications |
| `JobFeedback` | job_id, feedback_type, notes | Store, tools |
| `ApplicationPackage` | job, profile, resume_suggestions, cover_letter, intro_email | Store, app prep |

Enums: `JobStatus`, `DatePosted`, `FeedbackType`, `NotificationChannel`

### store.py

Abstract base class `JobStore` with two implementations:

- **`InMemoryJobStore`** — Dict-based, for testing. Data lost on restart.
- **`PostgresJobStore`** — `asyncpg` connection pool, pgvector extension for embeddings.

Key methods on the interface:
```
add_many(jobs)              → INSERT jobs
list_all(limit)             → SELECT jobs
update_status(job_id, st)   → UPDATE status
save_profile(profile)       → UPSERT user profile
get_default_profile()       → SELECT latest profile
update_job_embeddings(...)  → UPDATE with vector data
save_feedback(feedback)     → INSERT feedback
save_application_package()  → INSERT app package
```

`get_store(config)` is an async factory: creates the connection pool, runs `CREATE TABLE IF NOT EXISTS` / `CREATE EXTENSION IF NOT EXISTS vector`.

### providers.py

Abstract `JobIngestionProvider` with:

- **`SerpAPIProvider`** — Calls SerpAPI's Google Jobs endpoint via `httpx`. Parses job listings into `Job` dataclasses.
- **`MockJobProvider`** — Returns hardcoded test data.

`get_provider()` returns `SerpAPIProvider` if `SERPAPI_API_KEY` is set, otherwise `MockJobProvider`.

### ranking.py

Two layers:

1. **`EmbeddingService`** (abstract) → `AzureOpenAIEmbeddingService` or `MockEmbeddingService`
   - Calls Azure OpenAI's `text-embedding-3-small` model (1536 dimensions)
   - `embed(text) → float[]` and `embed_batch(texts) → float[][]`

2. **`RankingService`** — Orchestrates the ranking pipeline:
   - `embed_user_profile(profile)` — generates embedding from resume_text + skills
   - `embed_jobs(jobs)` — batch-embeds job descriptions
   - `rank_jobs(jobs, profile, top_k)` — composite scoring:
     - **Cosine similarity** (50%) — embedding dot product between profile and job
     - **Skill overlap** (25%) — set intersection of profile.skills vs job.skills
     - **Location match** (15%) — preferred_locations overlap
     - **Salary match** (10%) — job salary vs profile min_salary
   - Returns `list[RankedJob]` sorted by composite score, with justification strings

### resume_parser.py

`ResumeParser` — Extracts text from uploaded files:
- **PDF** → `pymupdf` (fitz) library
- **DOCX** → `python-docx` library
- **TXT** → plain read

After extraction, `parse_and_extract()` uses regex patterns to pull out name, email, skills, experience years, and current title. Returns a `ParsedResume` dataclass that can be converted to `UserProfile` via `.to_user_profile()`.

### notifications.py

`NotificationService` — Delivers job match notifications to configured channels:
- **Console** — Always enabled. Prints formatted output.
- **Email** — SMTP via `smtplib`
- **Teams** — Incoming webhook (HTTP POST with Adaptive Card JSON)
- **Slack** — Incoming webhook (HTTP POST with Block Kit JSON)

`send_job_matches(ranked_jobs, profile, title)` formats the notification body and sends to all enabled channels. Returns `dict[str, bool]` with success/failure per channel.

### application_prep.py

`ApplicationPrepService` — Generates tailored application materials using the LLM:

```
prepare_application(job, profile) → ApplicationPackage
  ├── generate_resume_suggestions(job, profile)  → list[str]  (diff-style bullet points)
  ├── generate_cover_letter(job, profile)        → str
  ├── generate_intro_email(job, profile)         → str
  └── find_recruiters(job)                       → list[dict] (Proxycurl API, if configured)
```

Each generation method constructs a detailed prompt with the job description and user profile, calls Azure OpenAI with `max_completion_tokens`, and parses the response. The `format_package_summary()` method creates a readable summary of the full package.

### tools.py

Module-level global state for the current user profile:

```python
_current_profile: Optional[UserProfile] = None

def get_current_profile() → Optional[UserProfile]
def set_current_profile(profile: UserProfile) → None
```

This acts as a session-scoped variable. It's set by `_ensure_profile()` on startup (from DB) or by the `set_user_profile` / `upload_resume` tools during conversation. Other tools read it via `get_current_profile()` to access the profile without it being a tool parameter.

> Note: This file also contains legacy `JobTools` and `RankingTools` classes that are no longer imported. The active tool implementations are `JobSearchTools` and `AppPrepTools` in `workflows.py`.

---

## Data Models

### Entity Relationship

```
UserProfile (1) ────── (*) Job           via ranking (embedding similarity)
     │                     │
     │                     │
     ▼                     ▼
ApplicationPackage (*)    JobFeedback (*)
```

### Embedding Pipeline

```
User sets profile (text or resume upload)
  → ResumeParser extracts text + skills
  → RankingService.embed_user_profile()
    → AzureOpenAIEmbeddingService.embed(resume_text + skills)
    → profile.embedding = float[1536]
  → store.save_profile(profile)

User searches for jobs
  → SerpAPIProvider.fetch_jobs()
  → store.add_many(jobs)
  → RankingService.embed_jobs(jobs)
    → AzureOpenAIEmbeddingService.embed_batch([job descriptions])
    → each job.embedding = float[1536]
  → store.update_job_embeddings([(id, embedding), ...])

User asks to rank
  → RankingService.rank_jobs(jobs, profile)
    → cosine_similarity(profile.embedding, job.embedding) for each job
    → add heuristic boosts (skills, location, salary)
    → sort by composite score
    → return RankedJob[] with justifications
```

---

## Key Design Decisions

### Why classifier-based routing instead of HandoffBuilder?

The Agent Framework provides a `HandoffBuilder` API for native agent-to-agent handoffs. We tried it extensively and found it **incompatible** with the `from_agent_framework(agent).run()` HTTP serving pattern:

- **Autonomous mode**: Specialist agents enter infinite tool-calling loops (hitting the 100-superstep limit) because they never "hand back" to the coordinator
- **Human-in-loop mode**: Returns `FunctionApprovalRequestContent` objects and empty `response.text` — the specialist's output never surfaces through `.as_agent()`

The classifier-based routing achieves the same multi-agent architecture with one additional lightweight LLM call per request (~100 tokens), while being fully compatible with WorkflowBuilder and HTTP serving.

### Why bound methods as tools instead of standalone functions?

Using `job_tools.search_jobs` (bound method on a `JobSearchTools` instance) gives each tool access to shared services via `self.store`, `self.provider`, etc. The alternative — standalone functions with global state — would make testing and dependency management much harder.

### Why a module-level global for user profile?

The user profile needs to be accessible from tool methods that don't have it as a parameter (e.g., `rank_saved_jobs` ranks against "the current profile"). A module-level global is the simplest approach for a single-user agent. For multi-tenant, this would be replaced with a context-scoped store keyed by session ID.

### Why WorkflowBuilder with a single node?

Even though we only have one executor, `WorkflowBuilder` provides:
- Proper `Executor` lifecycle management (factory pattern, graph resolution)
- `.as_agent()` to get `AgentProtocol` compliance
- Future extensibility — adding a second node (e.g., post-processing) would require only adding an edge
- Compatibility with `from_agent_framework()` for HTTP serving

### Why `max_completion_tokens` instead of `max_tokens`?

Azure OpenAI's `gpt-5.2` deployment uses the newer API parameter name. Using `max_tokens` raises an API error. This affects `ApplicationPrepService` where we set explicit generation limits.
