"""Classification accuracy evaluator.

Tests that the logprobs-based classifier correctly routes queries to the
right specialist agent (job_search_agent vs application_prep_agent).

Usage:
    python evals/eval_classifier.py            # run all golden queries
    python evals/eval_classifier.py --verbose   # show per-query detail
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

load_dotenv()

from job_agent.config import AppConfig
from job_agent.workflows import CLASSIFIER_INSTRUCTIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ClassifierResult:
    """Result of a single classification test."""

    query: str
    expected_agent: str
    predicted_agent: str
    confidence: float | None
    correct: bool
    alternatives: list[dict] = field(default_factory=list)


@dataclass
class ClassifierReport:
    """Aggregate classification report."""

    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    results: list[ClassifierResult] = field(default_factory=list)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CLASSIFIER ACCURACY REPORT",
            "=" * 60,
            f"Total queries:      {self.total}",
            f"Correct:            {self.correct}",
            f"Accuracy:           {self.accuracy:.1%}",
            f"Avg confidence:     {self.avg_confidence:.1f}%",
            "",
            "Confusion matrix:",
        ]
        # Build confusion matrix
        agents = sorted(
            {r.expected_agent for r in self.results}
            | {r.predicted_agent for r in self.results}
        )
        header = f"{'':>28s} | " + " | ".join(f"{a:>22s}" for a in agents)
        lines.append(header)
        lines.append("-" * len(header))
        for expected in agents:
            row_counts = []
            for predicted in agents:
                count = sum(
                    1
                    for r in self.results
                    if r.expected_agent == expected and r.predicted_agent == predicted
                )
                row_counts.append(f"{count:>22d}")
            lines.append(f"{'expected=' + expected:>28s} | " + " | ".join(row_counts))

        # Misclassified queries
        misses = [r for r in self.results if not r.correct]
        if misses:
            lines.append("")
            lines.append(f"MISCLASSIFIED ({len(misses)}):")
            for r in misses:
                conf = f" ({r.confidence:.1f}%)" if r.confidence else ""
                lines.append(
                    f"  ✗ [{r.expected_agent}→{r.predicted_agent}{conf}] "
                    f'"{r.query[:80]}"'
                )
        else:
            lines.append("\n✓ All queries classified correctly!")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------


async def evaluate_classifier(
    dataset_path: str = "evals/golden_dataset.jsonl",
    verbose: bool = False,
) -> ClassifierReport:
    """Run the classifier against the golden dataset and measure accuracy.

    This calls the actual Azure OpenAI logprobs classifier from workflows.py,
    using the same prompts and model the production system uses.
    """
    from openai import AsyncAzureOpenAI

    config = AppConfig.load()
    oc = config.azure_openai

    # Build the same client used in production
    if oc.api_key:
        client = AsyncAzureOpenAI(
            azure_endpoint=oc.endpoint,
            api_key=oc.api_key,
            api_version=oc.api_version,
        )
    else:
        from azure.identity.aio import DefaultAzureCredential

        client = AsyncAzureOpenAI(
            azure_endpoint=oc.endpoint,
            azure_ad_token_provider=DefaultAzureCredential(),
            api_version=oc.api_version,
        )

    # Load golden dataset
    dataset_path = str(Path(__file__).resolve().parent.parent / dataset_path)
    with open(dataset_path) as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    report = ClassifierReport(total=len(test_cases))
    confidences: list[float] = []

    for i, tc in enumerate(test_cases, 1):
        query = tc["query"]
        expected_agent = tc["expected_agent"]

        # Map expected_agent to classifier label
        expected_label = (
            "APP_PREP" if expected_agent == "application_prep_agent" else "JOB_SEARCH"
        )

        # Call classifier with exact same prompt as production
        classify_messages = [
            {"role": "system", "content": CLASSIFIER_INSTRUCTIONS},
            {"role": "user", "content": query},
        ]

        resp = await client.chat.completions.create(
            model=oc.deployment_name,
            messages=classify_messages,
            logprobs=True,
            top_logprobs=3,
            max_completion_tokens=150,
        )

        choice = resp.choices[0]
        predicted_label = (choice.message.content or "JOB_SEARCH").strip().upper()

        # Extract confidence
        confidence = None
        alternatives = []
        if choice.logprobs and choice.logprobs.content:
            first_token = choice.logprobs.content[0]
            confidence = round(math.exp(first_token.logprob) * 100, 1)
            for alt in first_token.top_logprobs:
                alternatives.append(
                    {
                        "token": alt.token,
                        "probability": round(math.exp(alt.logprob) * 100, 1),
                    }
                )

        # Map prediction back to agent name
        predicted_agent = (
            "application_prep_agent"
            if "APP_PREP" in predicted_label
            else "job_search_agent"
        )

        correct = predicted_agent == expected_agent
        if correct:
            report.correct += 1

        if confidence is not None:
            confidences.append(confidence)

        result = ClassifierResult(
            query=query,
            expected_agent=expected_agent,
            predicted_agent=predicted_agent,
            confidence=confidence,
            correct=correct,
            alternatives=alternatives,
        )
        report.results.append(result)

        if verbose:
            status = "✓" if correct else "✗"
            conf_str = f" ({confidence:.1f}%)" if confidence else ""
            print(f"  {status} [{i}/{report.total}] {predicted_label}{conf_str} — {query[:60]}")

    report.accuracy = report.correct / report.total if report.total > 0 else 0
    report.avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    await client.close()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print("Running classifier accuracy evaluation...")
    print(f"Dataset: evals/golden_dataset.jsonl\n")
    report = await evaluate_classifier(verbose=verbose)
    print(report.summary())

    # Save JSON report
    report_path = Path(__file__).resolve().parent / "results" / "classifier_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(
            {
                "total": report.total,
                "correct": report.correct,
                "accuracy": report.accuracy,
                "avg_confidence": report.avg_confidence,
                "results": [
                    {
                        "query": r.query,
                        "expected_agent": r.expected_agent,
                        "predicted_agent": r.predicted_agent,
                        "confidence": r.confidence,
                        "correct": r.correct,
                        "alternatives": r.alternatives,
                    }
                    for r in report.results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nJSON report saved to: {report_path}")

    # Exit with non-zero if accuracy < threshold
    threshold = 0.90
    if report.accuracy < threshold:
        print(f"\n⚠ BELOW THRESHOLD ({threshold:.0%})")
        sys.exit(1)
    else:
        print(f"\n✓ Above threshold ({threshold:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
