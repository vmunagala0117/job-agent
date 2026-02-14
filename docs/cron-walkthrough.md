# Cron Job — End-to-End Walkthrough

This document provides a detailed, step-by-step walkthrough of the automated
daily job search cron pipeline. It covers every component from the Azure
Functions timer trigger through agent execution, application material
generation, and result persistence.

---

## Architecture Overview

```
┌──────────────────────┐     POST /api/cron/daily-search
│  Azure Functions     │────────────────────────────────►┌──────────────────┐
│  Timer Trigger       │  (X-Cron-Key header)            │  FastAPI webapp   │
│  (6 AM UTC daily)    │◄────────────────────────────────│                  │
└──────────────────────┘     JSON response               │  cron_daily_     │
                                                         │  search()        │
                                                         └────────┬─────────┘
                                                                  │
                                           ┌──────────────────────┼──────────────────────┐
                                           │                      │                      │
                                     ┌─────▼──────┐       ┌──────▼───────┐       ┌──────▼───────┐
                                     │  Phase 1   │       │  Phase 2     │       │  Persist     │
                                     │  Search +  │       │  App Prep    │       │  SearchRun   │
                                     │  Rank +    │       │  (cover      │       │  record      │
                                     │  Notify    │       │  letters,    │       │  to DB       │
                                     │            │       │  resume      │       │              │
                                     └────────────┘       │  suggestions)│       └──────────────┘
                                                          └──────────────┘
```

**Key design decision:** The cron endpoint reuses the *same* multi-agent
pipeline as the chat UI. It sends synthetic user messages through
`agent.run()`, so all tool orchestration, ranking, notifications, and
application prep happen identically to a human-initiated session.

---

## Concrete Example

**Profile:** Vivek Ammun  
**Current role:** Senior Software Engineer  
**Skills:** Python, Azure, AI/ML, FastAPI, distributed systems  
**Desired titles:** Principal Engineer, Staff Engineer  
**Preferred locations:** Austin TX, Remote  
**Minimum salary:** $180,000  

---

## Step-by-Step Walkthrough

### Step 1 — Azure Function Timer Fires

**File:** `azure-functions/daily-search/function_app.py`

At **6:00 AM UTC** every day, the Azure Functions timer trigger fires. The
schedule is configured via the `CRON_SCHEDULE` environment variable (default:
`0 0 6 * * *` — NCRONTAB with seconds).

```python
@app.timer_trigger(schedule=SCHEDULE, arg_name="timer", run_on_startup=False)
def daily_job_search(timer: func.TimerRequest) -> None:
```

The function:
1. Reads `CRON_APP_URL` (e.g. `https://job-agent.azurewebsites.net`)
2. Reads `CRON_API_KEY` (shared secret)
3. Sends an HTTP POST with a 300-second timeout:

```http
POST https://job-agent.azurewebsites.net/api/cron/daily-search
Headers:
  X-Cron-Key: my-secret-key-123
  Content-Type: application/json
```

If `CRON_API_KEY` is not set, the function logs an error and aborts
without calling the endpoint.

---

### Step 2 — Authentication & Validation

**File:** `src/job_agent/webapp.py` — `_verify_cron_key()` and
`cron_daily_search()`

The FastAPI endpoint validates the request:

| Check | Failure Response |
|-------|------------------|
| `X-Cron-Key` header matches `CRON_API_KEY` env var | **401 Unauthorized** |
| Agent is initialised | **503 Service Unavailable** |
| Store (DB) is available | **503 Service Unavailable** |

If all checks pass, execution continues.

---

### Step 3 — Load All Profiles

```python
profiles = await store.list_profiles()
# → [UserProfile(name="Vivek Ammun", id="abc-123", ...)]
```

The endpoint iterates over every saved user profile. For each profile:

1. **Creates a `SearchRun` record** with status `RUNNING` and persists it
   to the `job_search_runs` database table
2. **Activates the profile** via `set_current_profile(profile)` so that all
   agent tools (ranking, notifications, app prep) see the correct user
   context — resume text, skills, embedding, preferences

---

### Step 4 — Phase 1: Synthetic Search Message

A **synthetic user message** is injected into a throwaway chat session:

```
"Run a daily automated job search based on my profile preferences.
 Use suggest_search_titles to get optimised keywords, then search_jobs
 for each keyword with max_results=30. Rank the results and send
 notifications for the top matches."
```

This message enters the **CoordinatorExecutor** classifier, which analyses
it and routes it to the **job_search_agent** (the specialist that owns the
12 search/rank/notify tools).

---

### Step 5 — Agent Autonomously Chains Tools

The job_search_agent (GPT-5.2) reads the message, reviews its instructions,
and autonomously decides which tools to call and in what order.

A typical execution trace:

| Step | Tool | What Happens |
|------|------|-------------|
| 5a | `get_profile()` | Loads Vivek's full profile — resume text, skills, desired titles, embedding |
| 5b | `suggest_search_titles()` | LLM analyses resume + desired titles → returns smart keywords:<br>`["Staff Python Engineer", "Principal Platform Engineer", "Senior AI/ML Engineer", "Staff Software Engineer Azure"]` |
| 5c | `search_jobs("Staff Python Engineer", location="Austin, TX", max_results=30)` | Hits SerpAPI → 28 jobs returned, each sorted by composite score (40% recency + 60% profile match) |
| 5d | `search_jobs("Principal Platform Engineer", location="Remote", max_results=30)` | 22 more jobs |
| 5e | *(repeats for remaining keywords)* | ~90 total jobs fetched and saved to the database |
| 5f | `rank_saved_jobs(top_k=10)` | Embedding-based ranking against Vivek's profile → top 10 scored |
| 5g | `send_job_notifications(top_k=5)` | Sends notifications (Teams / Slack / email) for the top 5 matches |

Each job in the agent's formatted output now includes the posting URL:

```
1. Staff Python Engineer at Stripe (Remote)
   | $190,000-$240,000 | Match: 92.0%
   | 🔗 https://stripe.com/careers/12345
   [ID: a1b2c3d4]

2. Principal Platform Engineer at Microsoft (Austin, TX)
   | $200,000-$260,000 | Match: 88.0%
   | 🔗 https://careers.microsoft.com/98765
   [ID: e5f6g7h8]
```

---

### Step 6 — Capture Top Matches

After Phase 1 completes, the cron endpoint collects results:

```python
recent_jobs = await store.list_all(limit=30)
top = [
    {
        "id": "a1b2c3d4",
        "title": "Staff Python Engineer",
        "company": "Stripe",
        "score": 0.92,
        "url": "https://stripe.com/careers/12345"
    },
    {
        "id": "e5f6g7h8",
        "title": "Principal Platform Engineer",
        "company": "Microsoft",
        "score": 0.88,
        "url": "https://careers.microsoft.com/98765"
    },
    # ... up to 10 scored jobs
]
```

These are stored in the `SearchRun.top_matches` field.

---

### Step 7 — Phase 2: Application Materials Generation

The cron filters top matches for jobs scoring above **30%** and selects
the best 5:

```python
prep_candidates = [j for j in recent_jobs[:10]
                   if j.score is not None and j.score > 0.3]
# → 9 jobs qualify (scores 0.92 → 0.35)
# Takes first 5: Stripe, Microsoft, Databricks, Anthropic, Snowflake
```

A **second synthetic message** is sent into the *same session* (the agent
retains context from Phase 1):

```
"For each of the following top job matches, prepare application
 materials (cover letter, resume suggestions, and intro email).
 Job IDs: a1b2c3d4, e5f6g7h8, i9j0k1l2, m3n4o5p6, q7r8s9t0"
```

The CoordinatorExecutor classifier routes this second message to the
**app_prep_agent** (the specialist that owns `prepare_application`,
`get_application_package`, and `find_recruiters`).

For each of the 5 jobs, the agent generates an `ApplicationPackage`:

| Material | Example (Stripe job) |
|----------|---------------------|
| **Cover Letter** | Tailored 3-paragraph letter highlighting Vivek's Python/FastAPI experience and Stripe's API-first culture |
| **Resume Suggestions** | 4 diff-style suggestions: "Add 'distributed systems' to skills", "Quantify Azure migration: 40% cost reduction", etc. |
| **Intro Email** | Short networking email template for the hiring manager |

Each package is saved to the `application_packages` database table. The
formatted summary also includes the job URL:

```
📋 APPLICATION PACKAGE
==================================================
Job: Staff Python Engineer at Stripe | 🔗 https://stripe.com/careers/12345
Status: draft
Created: 2026-02-14 06:01

📝 RESUME SUGGESTIONS
------------------------------
1. Add "distributed systems" to technical skills section...
2. Quantify Azure migration project: "Led migration of 12 services..."
3. ...

✉️ COVER LETTER DRAFT
------------------------------
Dear Hiring Manager,
...
```

The cron collects all package IDs created within the last 10 minutes:

```python
package_ids = ["pkg-aaa111", "pkg-bbb222", "pkg-ccc333", "pkg-ddd444", "pkg-eee555"]
```

---

### Step 8 — Persist SearchRun & Return Response

The `SearchRun` record is updated with final results:

```python
run.status           = SearchRunStatus.COMPLETED
run.jobs_found       = 90           # Total jobs fetched across all keywords
run.top_matches      = [...]        # Top 10 scored jobs with URLs
run.notification_channels = [...]   # 5 ApplicationPackage IDs
run.duration_ms      = 45200        # ~45 seconds total
```

The throwaway chat session is deleted, and the endpoint returns JSON to
the Azure Function:

```json
{
  "status": "ok",
  "profiles_processed": 1,
  "results": [
    {
      "profile": "Vivek Ammun",
      "status": "completed",
      "jobs_found": 90,
      "top_matches": 10,
      "packages_generated": 5,
      "duration_ms": 45200
    }
  ]
}
```

The Azure Function logs this summary and exits cleanly.

---

## How the User Views Results

| Channel | What the User Sees |
|---------|--------------------|
| **Teams / Slack / Email** (sent in Step 5g) | Notification with top 5 jobs — title, company, match score, and clickable 🔗 URL to the posting |
| **Chat UI → Cron Runs** (`GET /api/cron/runs`) | Browse historical cron runs — jobs found, top matches with URLs, generated package IDs, duration |
| **Chat UI → Ask the Agent** | "Show me my cover letter for the Stripe job" → agent retrieves `ApplicationPackage` and displays the tailored cover letter, resume suggestions, intro email, and 🔗 link to the job posting |

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| **One profile fails** | `SearchRun` marked `FAILED` with error message; cron continues to next profile |
| **Phase 2 fails** | Phase 1 results are preserved; `package_ids` remains empty; `SearchRun` still marked `COMPLETED` with `packages_generated: 0` |
| **Azure Function timeout** (300s) | The webapp continues processing in the background; `SearchRun` records are still persisted |
| **Auth failure** | 401 returned immediately; no profiles processed |
| **Agent or store not ready** | 503 returned immediately |

If the timer fires late (e.g. the Function App was cold), Azure Functions
sets `timer.past_due = True`. The function logs a warning but still
executes normally.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CRON_ENABLED` | No | `false` | Enable/disable the cron endpoint |
| `CRON_API_KEY` | Yes (for cron) | — | Shared secret for `X-Cron-Key` header |
| `CRON_SCHEDULE` | No | `0 0 6 * * *` | NCRONTAB schedule (6 fields with seconds) |
| `CRON_APP_URL` | Yes (for Azure Func) | `http://localhost:8080` | Base URL of the running webapp |

---

## Database Tables Involved

| Table | Role in Cron Pipeline |
|-------|----------------------|
| `user_profiles` | Source of profiles to iterate over |
| `jobs` | Stores fetched job listings (Phase 1) |
| `job_search_runs` | Tracks each cron execution — status, matches, packages, timing |
| `application_packages` | Stores generated cover letters, resume suggestions, intro emails (Phase 2) |

---

## Key Code Locations

| Component | File | Function / Line |
|-----------|------|-----------------|
| Azure Function trigger | `azure-functions/daily-search/function_app.py` | `daily_job_search()` |
| Cron endpoint | `src/job_agent/webapp.py` | `cron_daily_search()` |
| Auth check | `src/job_agent/webapp.py` | `_verify_cron_key()` |
| Cron run history | `src/job_agent/webapp.py` | `list_cron_runs()` |
| SearchRun model | `src/job_agent/models.py` | `class SearchRun` |
| Cron config | `src/job_agent/config.py` | `class CronConfig` |
| DB persistence | `src/job_agent/store.py` | `save_search_run()`, `update_search_run()` |
| DB schema | `scripts/init_db.py` | `job_search_runs` table |
| Composite sorting | `src/job_agent/workflows.py` | `_sort_jobs_composite()` |
| Smart keywords | `src/job_agent/workflows.py` | `suggest_search_titles()` |
| App prep service | `src/job_agent/application_prep.py` | `ApplicationPrepService` |
