# agentevals-grader-sdk

Lightweight SDK for building custom [agentevals](https://github.com/agentevals-dev/agentevals) graders.

A grader is a standalone program that scores agent traces. It reads `EvalInput` JSON from stdin and writes `EvalResult` JSON to stdout. This SDK provides the Python types and a `@grader` decorator that handles all the plumbing.

## Installation

```bash
pip install agentevals-grader-sdk
```

## Usage

```python
from agentevals_grader_sdk import grader, EvalInput, EvalResult

@grader
def my_grader(input: EvalInput) -> EvalResult:
    scores = []
    for inv in input.invocations:
        score = 1.0 if inv.final_response else 0.0
        scores.append(score)

    return EvalResult(
        score=sum(scores) / len(scores) if scores else 0.0,
        per_invocation_scores=scores,
    )
```

The `@grader` decorator turns your function into a runnable script -- just execute it with `python my_grader.py`. It reads JSON from stdin, calls your function, and writes the result to stdout.

## Types

- **`EvalInput`** -- input payload with `metric_name`, `threshold`, `config`, `invocations`, and optional `expected_invocations`
- **`EvalResult`** -- output payload with `score` (0.0-1.0), optional `status`, `per_invocation_scores`, and `details` (dict)
- **`InvocationData`** -- a single agent turn with `user_content`, `final_response`, `tool_calls`, and `tool_responses`

## Documentation

See the [custom graders documentation](https://github.com/agentevals-dev/agentevals/blob/main/docs/custom-graders.md) for the full protocol reference and examples in other languages.
