# Evaluation Guide

> How we measure whether the job-agent is actually doing a good job, and how to make it better over time.

---

## Why Evaluate an AI Agent?

An AI agent isn't like a normal function where you can check `assert output == expected`.
The model's responses vary, tools get called in different orders, and "good" is subjective.
Evaluation gives you **objective, repeatable signals** so you can answer questions like:

- Is the classifier routing queries to the right specialist?
- Are responses grounded in actual tool results (not hallucinated)?
- Did the quality get better or worse after a prompt change?
- Are users actually happy with the answers?

Without evaluation, you're flying blind — every prompt tweak is a guess.

---

## The Three Layers of Evaluation

Think of evaluation as three concentric rings, from cheapest/fastest to most expensive/valuable:

```
┌───────────────────────────────────────────┐
│  Layer 3: Human Feedback (production)     │  ← most valuable, slowest
│  ┌───────────────────────────────────┐    │
│  │ Layer 2: LLM-as-Judge (offline)   │    │  ← good signal, moderate cost
│  │ ┌───────────────────────────────┐ │    │
│  │ │ Layer 1: Code-Based (offline) │ │    │  ← cheapest, deterministic
│  │ └───────────────────────────────┘ │    │
│  └───────────────────────────────────┘    │
└───────────────────────────────────────────┘
```

| Layer | What | How | Speed | Cost |
|-------|------|-----|-------|------|
| **1. Code-based** | Classification accuracy, tool invocation, response length | Python assert/score functions | Seconds | Free |
| **2. LLM-as-judge** | Relevance, groundedness, coherence, fluency | A second LLM scores the agent's output | Minutes | $$$ (API calls) |
| **3. Human feedback** | Actual user satisfaction | Thumbs up/down in the UI | Continuous | Time |

We currently have all three layers implemented. Here's how each works.

---

## Layer 1: Code-Based Evaluators

These are deterministic Python functions that score agent behavior with no LLM calls.
They run in seconds and cost nothing.

### File: `evals/evaluators.py`

Three custom evaluators, each following the Azure AI Evaluation SDK pattern (`__init__` + `__call__`):

#### ResponseLengthEvaluator

Checks that the response is long enough to be useful.

| Score | Meaning |
|-------|---------|
| 1.0 | Response >= 100 chars (substantive) |
| 0.5 | Response between 50–100 chars (marginal) |
| 0.0 | Response < 50 chars (probably a failure) |

```python
from evals.evaluators import ResponseLengthEvaluator

evaluator = ResponseLengthEvaluator(min_chars=100)
result = evaluator(response="Here are 5 Python jobs in Seattle...")
# → {"response_length": 42, "response_length_score": 0.5}
```

#### ToolUsageEvaluator

Checks that the agent called the expected tools. Score = overlap / expected.

```python
from evals.evaluators import ToolUsageEvaluator

evaluator = ToolUsageEvaluator()
result = evaluator(
    expected_tools=["search_jobs", "rank_saved_jobs"],
    actual_tools=["search_jobs"]
)
# → {"tool_usage_score": 0.5, "tools_missing": ["rank_saved_jobs"]}
```

#### ClassificationEvaluator

Binary check: did the classifier route to the correct specialist agent?

```python
from evals.evaluators import ClassificationEvaluator

evaluator = ClassificationEvaluator()
result = evaluator(
    expected_agent="job_search_agent",
    actual_agent="application_prep_agent",
    classifier_confidence=92.3
)
# → {"classification_score": 0.0, "classification_correct": False}
```

---

## Layer 2: LLM-as-Judge Evaluators

These use Azure AI Evaluation SDK's built-in evaluators. A second LLM reads the agent's query, response, and context (tool results), then scores on a 1–5 scale.

### File: `evals/eval_response_quality.py`

Four evaluators run through the SDK's `evaluate()` API:

| Evaluator | What It Scores | Input Fields Used |
|-----------|---------------|-------------------|
| **RelevanceEvaluator** | Does the response address the user's question? | `query`, `response` |
| **GroundednessEvaluator** | Are claims backed by tool results, not hallucinated? | `response`, `context` |
| **CoherenceEvaluator** | Is the response well-structured and logical? | `query`, `response` |
| **FluencyEvaluator** | Is the language grammatically correct and natural? | `response` |

Each produces a score from 1 (worst) to 5 (best). The SDK aggregates across all rows
in the dataset and reports mean scores.

### How to run it

```bash
# Uses default dataset (evals/response_quality_dataset.jsonl)
python evals/eval_response_quality.py

# Use a custom dataset
python evals/eval_response_quality.py --dataset path/to/your_data.jsonl
```

Output goes to `evals/results/response_quality/` and a summary JSON.

---

## Classification Accuracy Suite

This is our highest-value code-based evaluation. It tests the **logprobs-based classifier**
(the routing brain of the entire multi-agent system) against a labeled golden dataset.

### File: `evals/eval_classifier.py`

What it does:
1. Loads 30 labeled queries from `evals/golden_dataset.jsonl`
2. Calls the exact same Azure OpenAI logprobs classifier used in production
3. Compares predicted agent vs expected agent
4. Generates a confusion matrix, per-query detail, and accuracy %
5. Exits non-zero if accuracy falls below **90% CI gate**

### How to run it

```bash
# Quick run
python evals/eval_classifier.py

# With per-query detail
python evals/eval_classifier.py --verbose
```

Sample output:
```
============================================================
CLASSIFIER ACCURACY REPORT
============================================================
Total queries:      30
Correct:            29
Accuracy:           96.7%
Avg confidence:     94.2%

Confusion matrix:
                             |    application_prep_agent |         job_search_agent
-----------------------------+--------------------------+--------------------------
 expected=application_prep   |                        9 |                        1
 expected=job_search         |                        0 |                       20

MISCLASSIFIED (1):
  ✗ [application_prep_agent→job_search_agent (67.3%)] "What's the job market like..."
============================================================
```

JSON report is saved to `evals/results/classifier_report.json`.

---

## Data Files

### `evals/golden_dataset.jsonl` — Classifier Ground Truth

30 labeled queries. Each line is a JSON object:

```json
{
  "query": "Find remote Python developer jobs paying over 150k",
  "expected_agent": "job_search_agent",
  "expected_tools": ["search_jobs"],
  "category": "job_search"
}
```

| Field | Purpose |
|-------|---------|
| `query` | The user's natural language input |
| `expected_agent` | Which specialist should handle it (`job_search_agent` or `application_prep_agent`) |
| `expected_tools` | Which tools the agent should call |
| `category` | Grouping tag for analysis |

**Current distribution:** 20 job_search queries, 10 application_prep queries.

### `evals/response_quality_dataset.jsonl` — LLM-as-Judge Ground Truth

6 annotated response examples. Each line:

```json
{
  "query": "Find remote Python developer jobs paying over 150k",
  "response": "I found 8 remote Python developer positions with salaries above $150,000...",
  "context": "Tool call: search_jobs(query='Python developer', remote_only=True, min_salary=150000)\nTool result: Found 8 jobs..."
}
```

| Field | Purpose |
|-------|---------|
| `query` | The user's question |
| `response` | The agent's actual response |
| `context` | Tool calls and results that ground the response |

---

## User Feedback (Layer 3)

The web UI provides thumbs up/down buttons on every response. Feedback is captured as:
- An in-memory store (accessible via `GET /api/feedback/analytics`)
- An OpenTelemetry span emitted to Application Insights

### Analytics Endpoint

```
GET /api/feedback/analytics
```

Returns:
```json
{
  "total_feedback": 42,
  "positive": 35,
  "negative": 7,
  "satisfaction_rate": 0.833,
  "daily_breakdown": [...],
  "recent_negative": [...]
}
```

### Querying in Application Insights

```kql
customEvents
| where name == "user_feedback"
| extend rating = tostring(customDimensions.rating)
| summarize
    thumbs_up   = countif(rating == "up"),
    thumbs_down = countif(rating == "down")
    by bin(timestamp, 1d)
| order by timestamp desc
```

---

## How to Add a New Evaluator

### Code-based (deterministic)

1. Add a class to `evals/evaluators.py`:

```python
class MyNewEvaluator:
    """One-line description of what this checks."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def __call__(self, *, response: str, **kwargs) -> dict:
        score = your_logic(response)
        return {"my_metric_score": score}
```

2. To use it with the SDK's `evaluate()` API, add it to `eval_response_quality.py`:

```python
result = evaluate(
    data=dataset_path,
    evaluators={
        "relevance": relevance,
        "my_metric": MyNewEvaluator(threshold=0.9),  # ← add here
    },
)
```

### LLM-as-judge (if you need nuanced scoring)

Use the Azure AI Evaluation SDK's built-in evaluators or create a prompt-based one.
See the [Azure AI Evaluation docs](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/evaluate-sdk) for available evaluators like `SimilarityEvaluator`, `ViolenceEvaluator`, etc.

---

## How to Expand the Golden Dataset

More test cases = more reliable evaluation. Here's how to add them:

1. Open `evals/golden_dataset.jsonl`
2. Add a new line with this structure:

```json
{"query": "Your new test query here", "expected_agent": "job_search_agent", "expected_tools": ["search_jobs"], "category": "job_search"}
```

3. Run `python evals/eval_classifier.py --verbose` to verify

**Tips for good test cases:**
- Include edge cases ("What should I wear to an interview?" — this is app_prep, not job_search)
- Include ambiguous queries that could go either way
- Balance the dataset across both agents
- Include queries with multiple expected tools

---

## Current State: What We Have vs. What's Missing

Here's an honest assessment of where we are:

### ✅ What Works Today

| Capability | Status | Notes |
|-----------|--------|-------|
| Classifier accuracy testing | Working | 30 queries, 90% CI gate, confusion matrix |
| LLM-as-judge scoring | Working | 4 evaluators via Azure AI Evaluation SDK |
| Custom code-based evaluators | Working | Length, tool usage, classification |
| User feedback collection | Working | Thumbs up/down → OTel spans + analytics API |
| Results persistence | Working | JSON reports saved to `evals/results/` |

### ⚠️ What's Missing (The Gap)

| Gap | Impact | Difficulty |
|-----|--------|------------|
| Evaluations are **offline only** | You must remember to run them manually | Medium |
| Feedback is **in-memory only** | Lost on restart, no history | Medium |
| Feedback is **disconnected** from evals | Thumbs-down doesn't trigger investigation | High |
| No **conversation persistence** | Can't replay or audit past sessions | High |
| No **regression detection** | Prompt changes aren't auto-tested | Medium |
| Golden dataset is **static** | Doesn't grow from real production queries | High |

---

## 🔶 Recommended Next Steps

> These are the improvements to revisit. They're ordered by impact and build on each other.

### Step 1: Persist Conversations to Database ⭐

**Why:** Without stored conversations, you can't replay, audit, or learn from real usage.

**What to build:**
- Add a `conversations` table to PostgreSQL:
  ```sql
  CREATE TABLE conversations (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      session_id TEXT NOT NULL,
      query TEXT NOT NULL,
      response TEXT NOT NULL,
      agent TEXT,              -- which specialist handled it
      tools_used JSONB,        -- list of tool names invoked
      classifier_confidence REAL,
      token_usage JSONB,       -- {input, output, total}
      feedback TEXT,           -- 'up', 'down', or NULL
      created_at TIMESTAMPTZ DEFAULT now()
  );
  ```
- Write to this table from `webapp.py` after every chat response
- Link feedback (thumbs up/down) to the conversation row by session + timestamp

**Payoff:** You now have a queryable history of every interaction, which unlocks Steps 2–4.

### Step 2: Auto-Diagnose Negative Feedback ⭐

**Why:** When a user gives thumbs-down, *something* went wrong — but what? Currently you'd have to guess. An LLM-as-judge can tell you.

**What to build:**
- When feedback = `down`, pull the conversation from the DB
- Run it through the LLM-as-judge evaluators (relevance, groundedness, coherence)
- Tag the conversation with the failure mode:
  - `wrong_agent` — classifier routed to the wrong specialist
  - `hallucination` — response wasn't grounded in tool results
  - `irrelevant` — response didn't address the question
  - `incomplete` — response was too short or missed key info
  - `tool_failure` — a tool returned an error or no results
- Store the diagnosis in a `diagnosis` column

**Payoff:** You know *why* users are unhappy, not just *that* they are.

### Step 3: Feed Failures into Golden Dataset ⭐

**Why:** Your golden dataset is static (30 hand-written queries). Real production failures are much more valuable test cases.

**What to build:**
- Add a review queue: `GET /api/eval/review` shows diagnosed negative feedback
- A human reviewer confirms or corrects the diagnosis
- One-click export: "Add to golden dataset" appends the query + correct label to `golden_dataset.jsonl`
- Over time, the golden dataset grows organically from real usage

**Payoff:** Your test suite gets smarter every week without you writing test cases.

### Step 4: CI/CD Gate (Run on PR) ⭐

**Why:** Prompt changes, model upgrades, and code refactors can silently break classification or response quality. You want to catch this *before* merging.

**What to build:**
- A GitHub Actions workflow (`.github/workflows/eval.yml`):
  ```yaml
  name: Evaluation Gate
  on: pull_request
  jobs:
    eval:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: pip install -r requirements.txt
        - run: python evals/eval_classifier.py
          env:
            AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
            AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
            AZURE_OPENAI_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_DEPLOYMENT_NAME }}
  ```
- The classifier eval already exits non-zero below 90% — so it works as a gate out of the box
- Optionally add response quality eval with score thresholds

**Payoff:** No regression ships to production undetected.

### Step 5: End-to-End Evaluation (Full Agent Run) ⭐

**Why:** The classifier eval tests routing, and LLM-as-judge tests pre-recorded responses. Neither tests the full pipeline: user query → classifier → specialist agent → tool calls → response.

**What to build:**
- A new script `evals/eval_e2e.py` that:
  1. Loads test queries from a dataset
  2. Sends them through the actual `workflow.run()` (not just the classifier)
  3. Captures the full response, tools called, and agent used
  4. Runs all evaluators (code-based + LLM-as-judge) on the live output
- This is slower and more expensive (every run calls real tools + LLM), but tests the real system

**Payoff:** Confidence that the whole pipeline works, not just individual parts.

### Step 6: A/B Prompt Testing ⭐

**Why:** When you change a system prompt, you want to compare old vs new objectively, not "it feels better."

**What to build:**
- Run the golden dataset against prompt version A and prompt version B
- Collect side-by-side scores for each evaluator
- Output a comparison table:
  ```
  Metric            Prompt A    Prompt B    Delta
  ────────────────────────────────────────────────
  Accuracy          96.7%       93.3%       -3.4%  ⚠️
  Relevance         4.5         4.7         +0.2
  Groundedness      4.2         4.6         +0.4  ✓
  Coherence         4.8         4.5         -0.3
  ```

**Payoff:** Data-driven prompt engineering instead of guesswork.

---

## Recommended Priority Order

Here's the order I'd tackle these in:

```
                    ┌──────────────┐
                    │  Step 1      │  Persist conversations
                    │  (foundation)│  ← everything else depends on this
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  Step 2   │  │  Step 4   │  │  Step 5   │
       │  Auto     │  │  CI/CD    │  │  E2E      │
       │  diagnose │  │  gate     │  │  eval     │
       └────┬──────┘  └──────────┘  └──────────┘
            │
            ▼
       ┌──────────┐
       │  Step 3   │  Feed failures into golden dataset
       │  (grows   │  ← requires Step 2's diagnosis
       │  dataset) │
       └────┬──────┘
            │
            ▼
       ┌──────────┐
       │  Step 6   │  A/B prompt testing
       │  (mature) │  ← most useful once golden dataset is large
       └──────────┘
```

**Steps 1 → 2 → 3** form the "feedback loop" — the most impactful improvement sequence.
**Step 4** is independent and can be done anytime.
**Step 5** is independent but benefits from a larger golden dataset (Step 3).
**Step 6** is the cherry on top once the foundation is solid.

---

## Quick Reference: Running Everything

```bash
# Classification accuracy (Layer 1, ~30 seconds, costs ~30 API calls)
python evals/eval_classifier.py --verbose

# Response quality (Layer 2, ~2 minutes, costs ~24 API calls)
python evals/eval_response_quality.py

# View feedback analytics (Layer 3, free, requires running server)
curl http://localhost:8080/api/feedback/analytics | python -m json.tool

# Check results
ls evals/results/
```

---

## FAQ

**Q: How often should I run evaluations?**
A: Run the classifier eval after any change to `CLASSIFIER_INSTRUCTIONS` in `workflows.py` or the model deployment. Run response quality eval after prompt changes to specialist agents. In the future, Step 4 automates this.

**Q: The LLM-as-judge scores seem low — is that bad?**
A: Scores are on a 1–5 scale. Anything above 3.5 is generally good. Below 3.0 warrants investigation. Remember: the judge LLM is scoring your agent LLM, so there's inherent variance.

**Q: How much does running evaluations cost?**
A: The classifier eval makes ~30 API calls (one per test case). The response quality eval makes ~24 calls (4 evaluators × 6 test rows). Total cost is typically under $1 per full run.

**Q: Can I run evaluations without Azure AI Evaluation SDK?**
A: Yes — the classifier eval (`eval_classifier.py`) and the custom evaluators (`evaluators.py`) only need `openai` and standard Python. The response quality eval (`eval_response_quality.py`) requires the SDK.

**Q: Where do results go?**
A: `evals/results/` — the classifier writes `classifier_report.json`, and the response quality eval writes to `response_quality/` subdirectory plus a summary JSON.
