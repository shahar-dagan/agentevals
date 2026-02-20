`trace-eval` scores agent behavior from OpenTelemetry traces without re-running the agent. It parses trace spans from Jaeger JSON format and evaluates them against golden eval sets using ADK's evaluation framework.

Unlike ADK's LocalEvalService, which couples agent execution with evaluation, trace-eval only handles scoring: it takes pre-recorded traces and compares them against expected behavior using metrics like tool trajectory matching, response quality, and LLM-based judgments.

The tool provides both a CLI for local dev work, scripting and CI pipelines, and a web UI for visual inspection, EvalSet creation and interactive evaluation. 

## Getting Started

Install dependencies using the Nix development environment (recommended) or uv:

```bash
# Using Nix (includes all dependencies)
nix develop .

# Or using uv directly
uv sync
```

Run a quick evaluation:

```bash
uv run trace-eval run samples/helm.json --eval-set samples/eval_set_helm.json -m tool_trajectory_avg_score
```

## CLI Usage

Score a single trace:

```bash
uv run trace-eval run samples/helm.json --eval-set samples/eval_set_helm.json -m tool_trajectory_avg_score
```

Score multiple traces at once:

```bash
uv run trace-eval run samples/helm.json samples/k8s.json --eval-set samples/eval_set_helm.json -m tool_trajectory_avg_score
```

Output as JSON for programmatic consumption:

```bash
uv run trace-eval run samples/helm.json --eval-set samples/eval_set_helm.json --output json
```

List available metrics:

```bash
uv run trace-eval list-metrics
```

## Web UI

The React-based UI provides visual trace inspection and interactive evaluation:

```bash
# Terminal 1: Start API server
uv run uvicorn trace_eval.api.app:app --reload --port 8000

# Terminal 2: Start UI dev server
cd ui && npm run dev
```

Open http://localhost:5173 to upload traces and eval sets, select metrics, and view results with interactive span trees and actual vs expected comparisons.

## Local Development

The project uses Nix for reproducible development environments. All dependencies (Python, Node.js, packages) are managed via `flake.nix`:

```bash
# Enter development shell
nix develop .

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_runner.py -v
```
