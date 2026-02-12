"""Response quality evaluation using Azure AI Evaluation SDK.

Measures:
  - Relevance: Does the response address the user's question?
  - Groundedness: Are claims backed by tool result context?
  - Coherence: Is the response well-structured and clear?
  - Fluency: Is the language grammatically correct and natural?

Uses built-in evaluators from azure-ai-evaluation, run through the
unified evaluate() API with automatic aggregation.

Usage:
    python evals/eval_response_quality.py
    python evals/eval_response_quality.py --dataset evals/my_data.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

load_dotenv()


def get_model_config():
    """Build AzureOpenAIModelConfiguration from existing env vars."""
    from azure.ai.evaluation import AzureOpenAIModelConfiguration

    return AzureOpenAIModelConfiguration(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )


def run_evaluation(dataset_path: str | None = None):
    """Run response quality evaluation using the evaluate() API."""
    from azure.ai.evaluation import (
        CoherenceEvaluator,
        FluencyEvaluator,
        GroundednessEvaluator,
        RelevanceEvaluator,
        evaluate,
    )

    model_config = get_model_config()

    # Resolve dataset path
    if dataset_path is None:
        dataset_path = str(
            Path(__file__).resolve().parent / "response_quality_dataset.jsonl"
        )

    # Output path for results
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "response_quality")

    print("=" * 60)
    print("RESPONSE QUALITY EVALUATION")
    print("=" * 60)
    print(f"Dataset:   {dataset_path}")
    print(f"Output:    {output_path}")
    print(f"Model:     {os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}")
    print(f"Endpoint:  {os.environ['AZURE_OPENAI_ENDPOINT']}")
    print()

    # Initialize built-in evaluators
    relevance = RelevanceEvaluator(model_config=model_config)
    groundedness = GroundednessEvaluator(model_config=model_config)
    coherence = CoherenceEvaluator(model_config=model_config)
    fluency = FluencyEvaluator(model_config=model_config)

    print("Running evaluators: relevance, groundedness, coherence, fluency...")
    print("This may take a minute...\n")

    # Run unified evaluation — the SDK handles parallel execution + aggregation
    result = evaluate(
        data=dataset_path,
        evaluators={
            "relevance": relevance,
            "groundedness": groundedness,
            "coherence": coherence,
            "fluency": fluency,
        },
        evaluator_config={
            "relevance": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "groundedness": {
                "column_mapping": {
                    "response": "${data.response}",
                    "context": "${data.context}",
                }
            },
            "coherence": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "fluency": {
                "column_mapping": {
                    "response": "${data.response}",
                }
            },
        },
        output_path=output_path,
    )

    # Display results
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    metrics = result.get("metrics", {})
    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {metric_name:>40s}: {value:.2f}")
        else:
            print(f"  {metric_name:>40s}: {value}")

    print()

    # Also save a concise summary
    summary_path = output_dir / "response_quality_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "dataset": dataset_path,
                "evaluators": ["relevance", "groundedness", "coherence", "fluency"],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Summary saved to: {summary_path}")
    print(f"Full results saved to: {output_path}/")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = None
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
    run_evaluation(dataset_path=dataset)
