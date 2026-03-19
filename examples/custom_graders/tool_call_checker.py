"""Example custom grader: checks that every invocation made at least one tool call.

Usage in eval_config.yaml:

    metrics:
      - name: tool_call_checker
        type: code
        path: ./examples/custom_graders/tool_call_checker.py
        threshold: 1.0
        config:
          min_tool_calls: 1
"""

from agentevals_grader_sdk import grader, EvalInput, EvalResult


@grader
def tool_call_checker(input: EvalInput) -> EvalResult:
    min_calls = input.config.get("min_tool_calls", 1)
    scores: list[float] = []

    for inv in input.invocations:
        if len(inv.tool_calls) >= min_calls:
            scores.append(1.0)
        else:
            scores.append(0.0)

    overall = sum(scores) / len(scores) if scores else 0.0
    return EvalResult(
        score=overall,
        per_invocation_scores=scores,
    )
