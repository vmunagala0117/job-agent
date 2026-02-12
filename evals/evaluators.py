"""Custom code-based evaluators for agent-specific metrics.

These evaluators follow the Azure AI Evaluation SDK pattern: a class with
``__init__()`` and ``__call__()`` methods.  They can be passed to the
``evaluate()`` API alongside the built-in evaluators.

Evaluators included:
  - ResponseLengthEvaluator:  flags responses that are too short to be useful.
  - ToolUsageEvaluator:       checks that expected tools were invoked.
  - ClassificationEvaluator:  checks that the classifier routed to the correct agent.
"""

from __future__ import annotations


class ResponseLengthEvaluator:
    """Check that responses are long enough to be useful.

    Scores:
      1.0  — response is >= min_chars (substantive)
      0.5  — response is between 50 and min_chars (marginal)
      0.0  — response is < 50 chars (too short / failure)
    """

    def __init__(self, min_chars: int = 100):
        self.min_chars = min_chars

    def __call__(self, *, response: str, **kwargs) -> dict:
        length = len(response or "")
        if length >= self.min_chars:
            score = 1.0
        elif length >= 50:
            score = 0.5
        else:
            score = 0.0
        return {
            "response_length": length,
            "response_length_score": score,
        }


class ToolUsageEvaluator:
    """Verify that the expected tools were invoked.

    Input data must contain:
      - expected_tools:  list[str] — tool names the query should trigger
      - actual_tools:    list[str] — tool names that were actually called

    Score = |intersection| / |expected|  (1.0 = perfect recall).
    If expected_tools is empty, score is 1.0 (no tools required).
    """

    def __init__(self):
        pass

    def __call__(
        self,
        *,
        expected_tools: str | list | None = None,
        actual_tools: str | list | None = None,
        **kwargs,
    ) -> dict:
        # Parse inputs (they may arrive as JSON strings from the JSONL)
        import json

        if isinstance(expected_tools, str):
            try:
                expected_tools = json.loads(expected_tools)
            except (json.JSONDecodeError, TypeError):
                expected_tools = [expected_tools] if expected_tools else []
        if isinstance(actual_tools, str):
            try:
                actual_tools = json.loads(actual_tools)
            except (json.JSONDecodeError, TypeError):
                actual_tools = [actual_tools] if actual_tools else []

        expected_set = set(expected_tools or [])
        actual_set = set(actual_tools or [])

        if not expected_set:
            return {"tool_usage_score": 1.0, "tools_expected": 0, "tools_matched": 0}

        matched = len(expected_set & actual_set)
        score = matched / len(expected_set)

        return {
            "tool_usage_score": score,
            "tools_expected": len(expected_set),
            "tools_matched": matched,
            "tools_missing": sorted(expected_set - actual_set),
            "tools_extra": sorted(actual_set - expected_set),
        }


class ClassificationEvaluator:
    """Check that the classifier routed to the correct agent.

    Input data must contain:
      - expected_agent: str  — e.g., "job_search_agent"
      - actual_agent:   str  — e.g., "application_prep_agent"

    Returns score 1.0 if they match, 0.0 otherwise, plus confidence
    if provided.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        *,
        expected_agent: str = "",
        actual_agent: str = "",
        classifier_confidence: float | None = None,
        **kwargs,
    ) -> dict:
        correct = expected_agent.strip() == actual_agent.strip()
        result = {
            "classification_score": 1.0 if correct else 0.0,
            "classification_correct": correct,
            "expected_agent": expected_agent,
            "actual_agent": actual_agent,
        }
        if classifier_confidence is not None:
            result["classifier_confidence"] = classifier_confidence
        return result
