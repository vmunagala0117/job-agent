# Job Agent

A production-grade AI job search assistant built with **Microsoft Agent Framework** and **Azure OpenAI**. A multi-agent system that searches for jobs, ranks them against your profile, and generates tailored application materials — all through natural language conversation.

## Quick Start

```powershell
# 1. Clone and setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# 2. Configure (copy .env.example → .env, fill in Azure OpenAI credentials)

# 3. Run the web app
python -m job_agent.webapp
# Open http://localhost:8000
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Web App                         │
│  Chat UI • Profile Management • Resume Upload • Feedback  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              CoordinatorExecutor                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Classifier (logprobs-based, confidence scoring)   │  │
│  │  → JOB_SEARCH (98.6%)  or  APP_PREP (91.4%)       │  │
│  └──────────┬──────────────────────┬─────────────────┘  │
│             │                      │                     │
│             ▼                      ▼                     │
│  ┌──────────────────┐   ┌────────────────────┐          │
│  │ Job Search Agent  │   │ App Prep Agent     │          │
│  │ (11 tools)        │   │ (3 tools)          │          │
│  │ • search_jobs     │   │ • analyze_job_fit  │          │
│  │ • get_profile     │   │ • prepare_app      │          │
│  │ • rank_saved      │   │ • get_package      │          │
│  │ • list_saved      │   └────────────────────┘          │
│  │ • get_details     │                                   │
│  │ • set_profile     │                                   │
│  │ • upload_resume   │                                   │
│  │ • send_notifs     │                                   │
│  │ • feedback        │                                   │
│  │ • mark_applied    │                                   │
│  │ • mark_rejected   │                                   │
│  └──────────────────┘                                    │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   Shared Services                         │
│  • PostgreSQL + pgvector    • Azure OpenAI (gpt-5.2)      │
│  • RankingService           • EmbeddingService             │
│  • NotificationService      • ResumeParser                 │
│  • ApplicationPrepService   • OpenTelemetry + App Insights │
└──────────────────────────────────────────────────────────┘
```

The system uses **classifier-based routing** — a lightweight LLM call with logprobs classifies each request as `JOB_SEARCH` or `APP_PREP`, then the appropriate specialist agent runs with the full conversation history and its own tool set.

## Features

| Category | Capabilities |
|----------|-------------|
| **Multi-Agent System** | Coordinator with classifier-based routing, logprobs confidence scoring, two specialist agents (14 tools total) |
| **Job Search** | SerpAPI Google Jobs (aggregates LinkedIn, Indeed, Glassdoor), embedding-based ranking with heuristic boosts |
| **Profile Management** | Web-based profile form, PDF/DOCX resume upload with auto-extraction of skills/experience/title |
| **Application Prep** | LLM-generated resume diff suggestions, cover letters, intro emails, job fit analysis |
| **Notifications** | Console, email (SMTP), Microsoft Teams (webhooks), Slack (webhooks) |
| **Feedback** | Thumbs up/down on responses, per-job feedback (good fit, not relevant, tailor resume, etc.) |
| **Persistence** | PostgreSQL + pgvector for jobs, profiles, feedback, application packages |
| **Observability** | OpenTelemetry → Azure Application Insights, structured audit spans, classifier confidence tracking |
| **Web UI** | Chat interface, profile modal, resume upload, collapsible trace panel, feedback buttons |

## Project Structure

```
src/job_agent/
├── webapp.py            # FastAPI web application (primary entry point)
├── workflows.py         # Multi-agent orchestration (Coordinator + Specialists)
├── config.py            # Configuration (Azure OpenAI, SerpAPI, PostgreSQL)
├── clients.py           # Azure OpenAI client factory
├── models.py            # Data models (Job, UserProfile, RankedJob, etc.)
├── store.py             # Storage (InMemory / PostgreSQL with pgvector)
├── providers.py         # Job ingestion (SerpAPI / Mock)
├── ranking.py           # Embedding + heuristic ranking
├── resume_parser.py     # PDF/DOCX parsing and skill extraction
├── notifications.py     # Notification delivery (email/Teams/Slack/console)
├── application_prep.py  # Application material generation (resume, cover letter, email)
├── tools.py             # Shared profile state
├── server.py            # Headless Agent Server SDK entry point
└── static/              # Web UI (HTML/CSS/JS)
    ├── index.html
    ├── css/style.css
    └── js/app.js

docs/
├── TECHNICAL.md         # Internal code flow and architecture details
└── kql-queries.kql      # 14 ready-to-run Application Insights queries

scripts/
├── init_db.py           # Database schema initialization
├── test_e2e.py          # End-to-end integration test
├── test_cli.py          # Interactive CLI test
├── test_routing.py      # Multi-agent routing test
├── test_workflow.py     # Workflow smoke test
├── test_notifications.py # Notification/feedback/prep test
└── test_otel.py         # OpenTelemetry trace test
```

## Prerequisites

- **Python 3.10+** (tested with 3.13)
- **Azure OpenAI** with a chat deployment (gpt-4o / gpt-5.2) + embeddings (text-embedding-3-small)
- **PostgreSQL + pgvector** for persistent storage (optional — falls back to in-memory)
- **SerpAPI** account for real job data (optional — falls back to mock data)
- **Auth**: `az login` for `DefaultAzureCredential`, or set `AZURE_OPENAI_API_KEY`

## Setup

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required — Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=<chat-deployment>
AZURE_OPENAI_API_KEY=<optional-if-using-managed-identity>

# Optional — Real job data
SERPAPI_API_KEY=<your-serpapi-key>

# Optional — Persistent storage
DATABASE_URL=postgresql://user:password@host:5432/job_agent

# Optional — Observability
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...

# Optional — Recruiter search
PROXYCURL_API_KEY=<your-proxycurl-key>
```

### 3. Database (Optional)

```bash
# Docker
docker run -d --name job-agent-db \
  -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_DB=job_agent \
  -p 5432:5432 pgvector/pgvector:pg16

# Initialize schema
python scripts/init_db.py
```

For production, use **Azure Database for PostgreSQL Flexible Server** with the pgvector extension enabled.

### 4. Run

```bash
# Web application (recommended)
python -m job_agent.webapp

# Or headless server (Agent Server SDK)
python -m job_agent.server --server
```

Open [http://localhost:8000](http://localhost:8000) for the web UI.

## Web API Reference

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Chat UI |
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Send message, get response |
| `POST` | `/api/upload-resume` | Upload PDF/DOCX resume |
| `GET` | `/api/profiles` | List all profiles |
| `GET` | `/api/profiles/{id}` | Get profile details |
| `POST` | `/api/profiles/save` | Create/update profile |
| `POST` | `/api/profiles/select` | Switch active profile |
| `POST` | `/api/feedback` | Thumbs up/down on response |
| `GET` | `/api/feedback` | List feedback entries |
| `GET` | `/api/traces` | Get trace log for session |
| `POST` | `/api/chat/reset` | Reset conversation |

## Usage Examples

The agent understands natural language:

```
# Job Search
"Search for Python developer jobs in Seattle"
"Find remote machine learning engineer positions"
"Look for data scientist jobs with salary above $150,000"
"Search for jobs based on my preferences"

# Profile & Ranking
"Show me my profile"
"Rank my saved jobs"
"Show me the best job matches"

# Application Prep
"Generate a cover letter for this role"
"Draft an intro email to the hiring manager"
"Analyze my fit for this position"

# Job Management
"List my saved jobs"
"Show details for job abc123"
"Mark job abc123 as applied"
```

## How Ranking Works

1. **Profile Embedding** — Resume + skills embedded via Azure OpenAI (text-embedding-3-small, 1536d)
2. **Job Embedding** — Title + description + skills embedded
3. **Cosine Similarity** (50%) — Semantic match between profile and job
4. **Skill Overlap** (25%) — Set intersection of skills
5. **Location Match** (15%) — Preferred locations overlap
6. **Salary Match** (10%) — Min salary comparison
7. **Composite Score** — Weighted 0–100%

## Observability

The system emits structured OpenTelemetry spans to Azure Application Insights:

- **Classifier spans** — Intent, confidence %, alternatives, user message
- **Tool call spans** — Tool name, arguments, result preview, agent attribution
- **Feedback spans** — Rating (up/down), response preview
- **Token usage** — Input/output/total per request

See [docs/kql-queries.kql](docs/kql-queries.kql) for 14 ready-to-run KQL queries.

## Configuration Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Chat model deployment name | Yes |
| `AZURE_OPENAI_API_KEY` | API key (leave empty for managed identity) | No |
| `AZURE_OPENAI_API_VERSION` | API version (default: `2024-05-01-preview`) | No |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`) | No |
| `SERPAPI_API_KEY` | SerpAPI key for Google Jobs | No |
| `DATABASE_URL` | PostgreSQL connection URL | No |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Azure Monitor connection string | No |
| `PROXYCURL_API_KEY` | Proxycurl API for recruiter search | No |

## Documentation

- **[docs/TECHNICAL.md](docs/TECHNICAL.md)** — Internal code flow, request lifecycle, module reference, design decisions
- **[docs/kql-queries.kql](docs/kql-queries.kql)** — Application Insights KQL queries for monitoring

## Roadmap

- [x] Multi-agent orchestration (Coordinator → Job Search + App Prep)
- [x] FastAPI web application with chat UI
- [x] Profile management with resume upload
- [x] Persistent storage (PostgreSQL + pgvector)
- [x] Embedding-based job ranking with heuristic boosts
- [x] Resume parsing (PDF/DOCX with skill extraction)
- [x] Notification delivery (email/Teams/Slack/console)
- [x] User feedback (thumbs up/down + per-job feedback)
- [x] Application prep (resume suggestions, cover letters, intro emails)
- [x] Enterprise monitoring (OpenTelemetry → Azure Application Insights)
- [x] Classifier confidence scoring (logprobs)
- [x] Structured audit logging (tool-level OTel spans)
- [ ] Recruiter search integration
- [ ] Scheduled job ingestion (cron-based auto-search)
- [ ] Evaluation and tracing (pytest-agent-evals)
