"""Evaluation harness for the Job Agent.

Modules:
  - eval_classifier:        Classification accuracy (logprobs → correct agent)
  - eval_response_quality:  LLM-as-judge scoring (relevance, groundedness, coherence, fluency)
  - evaluators:             Custom code-based evaluators (response length, tool usage, classification)
"""
