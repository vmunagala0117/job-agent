# OpenTelemetry Observability Guide

A beginner-friendly guide to how OpenTelemetry (OTel) works in the Job Agent project — how metrics, traces, and logs are collected and shipped to Azure Application Insights.

---

## What Is OpenTelemetry?

Think of OpenTelemetry as a **flight recorder for your application**. Just like an airplane's black box records everything that happens during a flight, OTel records what happens inside your code — what was called, how long it took, what the inputs/outputs were.

## The Three Concepts

OTel collects three types of data:

| Concept | Analogy | In This Project |
|---------|---------|-----------------|
| **Traces** (spans) | Security camera footage — shows *what happened* step by step | "Classifier ran → picked JOB_SEARCH at 98.6% → Job Search Agent ran → called `search_jobs` → returned 8 jobs" |
| **Metrics** | A dashboard gauge — shows *counts and measurements* | Token usage (input/output), response times |
| **Logs** | A diary — text messages about events | `[TRACE] Classifier → JOB_SEARCH (confidence: 98.6%)` |

---

## How It's Wired Up (Step by Step)

### Step 1: Setup at Startup

When the app starts (`webapp.py`, lines 177–198), the following happens:

```
App starts up
    │
    ▼
Check: is APPLICATIONINSIGHTS_CONNECTION_STRING set in .env?
    │
    ├── YES → Create 3 exporters:
    │         • AzureMonitorTraceExporter   (sends spans)
    │         • AzureMonitorMetricExporter   (sends metrics)
    │         • AzureMonitorLogExporter      (sends logs)
    │
    └── NO  → Use default console exporters (prints to terminal)
    │
    ▼
Call configure_otel_providers(exporters=...)
    │
    ▼
OTel is now active globally — any code can create spans
```

Here's the actual code:

```python
# webapp.py — inside the lifespan() startup function

from agent_framework.observability import configure_otel_providers

exporters = []
ai_conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
if ai_conn_str:
    from azure.monitor.opentelemetry.exporter import (
        AzureMonitorLogExporter,
        AzureMonitorMetricExporter,
        AzureMonitorTraceExporter,
    )
    exporters.append(AzureMonitorTraceExporter(connection_string=ai_conn_str))
    exporters.append(AzureMonitorMetricExporter(connection_string=ai_conn_str))
    exporters.append(AzureMonitorLogExporter(connection_string=ai_conn_str))

configure_otel_providers(exporters=exporters if exporters else None)
```

`configure_otel_providers()` is a helper from the Agent Framework. It does the plumbing: registers a **TracerProvider** (manages spans), a **MeterProvider** (manages metrics), and a **LoggerProvider** (manages logs). After this call, any code anywhere in the app can create spans and they'll automatically flow to the configured destination.

### Step 2: Creating a Tracer

At the top of `workflows.py`:

```python
from opentelemetry import trace

_tracer = trace.get_tracer("job_agent.workflows")
```

This creates a **tracer** — think of it as a named pen. Everything this tracer writes is labeled `"job_agent.workflows"` so you can filter by source later in Application Insights.

A second tracer is created in `webapp.py` for feedback events:

```python
tracer = trace.get_tracer("job_agent.feedback")
```

### Step 3: Creating Spans

Spans are created with Python `with` blocks:

```python
with _tracer.start_as_current_span("classifier") as span:
    # Everything inside this block is "the classifier span"
    # It records: start time, end time, and any attributes you attach

    span.set_attribute("classifier.intent", "JOB_SEARCH")
    span.set_attribute("classifier.confidence_pct", 98.6)
```

**A span is like a stopwatch with a notepad.** When you enter the `with` block, the stopwatch starts. You can write notes (attributes) on the notepad. When you exit the block, the stopwatch stops. The complete record (start time, end time, notes) gets sent to the exporter.

### Step 4: What Gets Recorded

Here's every span the app creates during a single user request:

```
User sends "Find Python jobs in Seattle"
    │
    ▼
┌─ Span: "classifier" ─────────────────────────────┐
│  Started: 12:00:00.000                            │
│  Ended:   12:00:00.450                            │
│  Attributes:                                      │
│    classifier.intent = "JOB_SEARCH"               │
│    classifier.confidence_pct = 98.6               │
│    classifier.alternatives = "JOB=98.6%, APP=1.2%"│
│    classifier.user_message = "Find Python jobs..." │
└───────────────────────────────────────────────────┘
    │
    ▼
┌─ Span: "specialist_agent" ────────────────────────┐
│  Started: 12:00:00.451                            │
│  Ended:   12:00:03.200                            │
│  Attributes:                                      │
│    agent.name = "job_search_agent"                 │
│    agent.intent = "JOB_SEARCH"                    │
└───────────────────────────────────────────────────┘
    │
    ▼
┌─ Span: "tool_call" ──────────────────────────────┐
│  Attributes:                                      │
│    tool.name = "search_jobs"                      │
│    tool.arguments = "query='Python', loc='Seattle'"│
│    tool.result_preview = "Found 8 jobs matching..."│
│    tool.agent = "job_search_agent"                │
│    audit.type = "tool_invocation"                 │
└───────────────────────────────────────────────────┘
```

When a user clicks thumbs-down, a separate span is emitted:

```
┌─ Span: "response_feedback" ──────────────────────┐
│  feedback.session_id = "abc-123"                  │
│  feedback.message_index = 3                       │
│  feedback.rating = "down"                         │
│  feedback.comment = "Showed wrong location"       │
└───────────────────────────────────────────────────┘
```

---

## Data Flow: From Code to Azure Portal

```
Your App Code
    │
    │  creates spans, metrics, logs
    ▼
┌──────────────────────┐
│  OTel SDK (in-memory)│  ← Collects everything, batches it
└──────────┬───────────┘
           │
           │  periodically flushes (every few seconds)
           ▼
┌──────────────────────────────────┐
│  AzureMonitorTraceExporter       │  ← Converts to Azure format
│  AzureMonitorMetricExporter      │    and sends over HTTPS
│  AzureMonitorLogExporter         │
└──────────┬───────────────────────┘
           │
           │  HTTPS POST to Azure
           ▼
┌──────────────────────────────────┐
│  Azure Application Insights      │  ← Stores everything
│  (Azure Monitor)                 │    Query with KQL
│                                  │    in Azure Portal
└──────────────────────────────────┘
```

The key insight: **your code never talks to Azure directly.** It just creates spans and attaches attributes. The OTel SDK batches them in memory and the exporter ships them to Azure in the background. If the connection string isn't set, everything just goes to the console — zero crashes, zero impact on the app.

---

## Where Each Piece Lives in Code

| What | File | Lines | Does What |
|------|------|-------|-----------|
| OTel setup + exporters | `webapp.py` | 177–198 | Creates exporters, calls `configure_otel_providers()` |
| Tracer creation | `workflows.py` | 51 | `_tracer = trace.get_tracer("job_agent.workflows")` |
| Classifier span | `workflows.py` | 903–935 | Records intent, confidence, alternatives |
| Agent routing span | `workflows.py` | 947–958 | Records which agent handled the request |
| Tool call audit spans | `workflows.py` | 996–1003 | One span per tool: name, args, result preview |
| Feedback span | `webapp.py` | 611–619 | Records thumbs up/down events |

---

## The Agent Framework's Role

`configure_otel_providers()` from `agent_framework.observability` does something important behind the scenes: it also instruments the **Agent Framework itself**. This means the framework automatically creates spans for internal operations (LLM calls, tool execution) without you writing any code. The custom spans in `workflows.py` (classifier, audit, tool calls) are *in addition to* the framework's built-in instrumentation.

---

## Span Attribute Reference

### Classifier Span (`"classifier"`)

| Attribute | Type | Example | Purpose |
|-----------|------|---------|---------|
| `classifier.intent` | string | `"JOB_SEARCH"` | Which category the classifier chose |
| `classifier.confidence_pct` | float | `98.6` | How confident the classifier was (0–100) |
| `classifier.alternatives` | string | `"JOB=98.6%, APP=1.2%"` | Top-3 logprobs alternatives |
| `classifier.user_message` | string | `"Find Python jobs..."` | First 200 chars of user input |

### Agent Span (`"specialist_agent"`)

| Attribute | Type | Example | Purpose |
|-----------|------|---------|---------|
| `agent.name` | string | `"job_search_agent"` | Which specialist handled the request |
| `agent.intent` | string | `"JOB_SEARCH"` | The classified intent |

### Tool Call Span (`"tool_call"`)

| Attribute | Type | Example | Purpose |
|-----------|------|---------|---------|
| `tool.name` | string | `"search_jobs"` | Which tool was invoked |
| `tool.arguments` | string | `"query='Python', loc='Seattle'"` | Tool input (first 500 chars) |
| `tool.result_preview` | string | `"Found 8 jobs matching..."` | Tool output (first 300 chars) |
| `tool.agent` | string | `"job_search_agent"` | Which agent invoked this tool |
| `audit.type` | string | `"tool_invocation"` | Always `"tool_invocation"` |

### Feedback Span (`"response_feedback"`)

| Attribute | Type | Example | Purpose |
|-----------|------|---------|---------|
| `feedback.session_id` | string | `"abc-123"` | Session that generated the response |
| `feedback.message_index` | int | `3` | Which message in the session |
| `feedback.rating` | string | `"up"` or `"down"` | User's rating |
| `feedback.comment` | string | `"Wrong location"` | Optional user comment |

---

## Querying in Azure Portal

Once spans arrive in Application Insights, you query them with **KQL** (Kusto Query Language) in the Azure Portal under **Logs**.

### Example: Find all low-confidence classifications

```kql
dependencies
| where name == "classifier"
| extend confidence = toreal(customDimensions["classifier.confidence_pct"])
| where confidence < 80
| project timestamp, confidence,
         customDimensions["classifier.intent"],
         customDimensions["classifier.user_message"]
| order by timestamp desc
```

### Example: Show all tool calls in the last hour

```kql
dependencies
| where name == "tool_call"
| where timestamp > ago(1h)
| project timestamp,
         customDimensions["tool.name"],
         customDimensions["tool.arguments"],
         customDimensions["tool.agent"]
| order by timestamp desc
```

### Example: Show all thumbs-down feedback

```kql
dependencies
| where name == "response_feedback"
| extend rating = tostring(customDimensions["feedback.rating"])
| where rating == "down"
| project timestamp,
         customDimensions["feedback.session_id"],
         customDimensions["feedback.comment"]
| order by timestamp desc
```

For a complete set of 14 ready-to-run KQL queries, see [`docs/kql-queries.kql`](kql-queries.kql).

---

## What Happens Without Application Insights?

If `APPLICATIONINSIGHTS_CONNECTION_STRING` is not set:

- **No crash** — the OTel setup still runs, using console exporters
- **Spans still work** — they print to stdout instead of Azure
- **All `_tracer.start_as_current_span()` calls still execute** — they just don't go anywhere persistent
- **The `[TRACE]` log lines** in the terminal are Python `logging` output, not OTel spans — they always appear regardless of OTel configuration

This is by design. OTel is purely additive — your app works identically with or without it.

---

## Common Questions

**Q: Does OTel slow down my app?**
No. Span creation is microseconds. The SDK batches and sends data in a background thread, never blocking your request.

**Q: What's the difference between the `[TRACE]` log lines and OTel spans?**
The `[TRACE]` log lines (e.g., `[TRACE] Classifier → JOB_SEARCH`) are standard Python `logging.info()` calls. They appear in the terminal and in the UI's trace panel. OTel spans are a separate, structured data format that is sent to Application Insights. Both record similar information, but spans have structured attributes that are queryable with KQL, while log lines are plain text.

**Q: Can I add new spans?**
Yes. Import the tracer and wrap any code block:

```python
from opentelemetry import trace

tracer = trace.get_tracer("my_module")

with tracer.start_as_current_span("my_operation") as span:
    span.set_attribute("my.attribute", "value")
    # ... your code here ...
```

The span will automatically flow to whichever exporter is configured.

**Q: How do I see spans locally without Azure?**
Run the app normally — `configure_otel_providers()` with no exporters uses `ConsoleSpanExporter`, which prints spans to stdout. You'll see JSON-formatted span data in your terminal.
