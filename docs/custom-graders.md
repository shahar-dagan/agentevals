# Custom Graders

Custom graders let you score agent traces with your own logic. A grader is any program that reads `EvalInput` JSON from stdin and writes `EvalResult` JSON to stdout. This simple protocol means you can write graders in Python, JavaScript/TypeScript, or any language that can read/write JSON.

## Quick Start

### 1. Scaffold a grader

```bash
agentevals grader init my_grader
```

This creates a directory with boilerplate code and a `grader.yaml` manifest:

```
my_grader/
├── my_grader.py     # scoring logic (implement your checks here)
└── grader.yaml      # metadata manifest
```

You can also specify a language:

```bash
agentevals grader init my_grader --runtime js    # JavaScript
agentevals grader init my_grader.ts              # TypeScript (inferred from extension)
```

### 2. Install the SDK (Python only)

```bash
pip install agentevals-grader-sdk
```

### 3. Write a grader

```python
# graders/response_quality.py
from agentevals_grader_sdk import grader, EvalInput, EvalResult

@grader
def response_quality(input: EvalInput) -> EvalResult:
    scores = []
    for inv in input.invocations:
        if not inv.final_response:
            scores.append(0.0)
        elif len(inv.final_response.strip()) < input.config.get("min_length", 10):
            scores.append(0.5)
        else:
            scores.append(1.0)

    return EvalResult(
        score=sum(scores) / len(scores) if scores else 0.0,
        per_invocation_scores=scores,
    )
```

The `@grader` decorator handles all the stdin/stdout plumbing. Your function receives an `EvalInput` and returns an `EvalResult`.

### 3. Add it to your eval config

```yaml
# eval_config.yaml
metrics:
  - tool_trajectory_avg_score   # built-in metric

  - name: response_quality      # your custom grader
    type: code
    path: ./graders/response_quality.py
    threshold: 0.7
    config:
      min_length: 20
```

### 4. Run

```bash
agentevals run traces/my_trace.json \
  --config eval_config.yaml \
  --eval-set eval_set.json
```

## Eval Config Reference

Each custom grader entry in the `metrics` list uses the following fields:

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | yes | | Unique name for the grader (used in output) |
| `type` | yes | | `code` for local code files |
| `path` | yes | | Path to the grader file (`.py`, `.js`, or `.ts`) |
| `threshold` | no | `0.5` | Score at or above this value means PASSED |
| `timeout` | no | `30` | Subprocess timeout in seconds |
| `config` | no | `{}` | Arbitrary key-value pairs passed to the grader |

## Protocol

Every grader — regardless of language — communicates via the same JSON protocol over stdin/stdout.

### Input (`EvalInput`)

```json
{
  "metric_name": "response_quality",
  "threshold": 0.7,
  "config": { "min_length": 20 },
  "invocations": [
    {
      "invocation_id": "inv-001",
      "user_content": "What is 2+2?",
      "final_response": "The answer is 4.",
      "tool_calls": [
        { "name": "calculator", "args": { "expression": "2+2" } }
      ],
      "tool_responses": [
        { "name": "calculator", "output": "4" }
      ]
    }
  ],
  "expected_invocations": null
}
```

| Field | Type | Description |
|---|---|---|
| `metric_name` | string | Name of this grader |
| `threshold` | float | Pass/fail threshold |
| `config` | object | User-provided config from the YAML |
| `invocations` | array | Agent turns to evaluate |
| `expected_invocations` | array or null | Golden reference turns (from eval set) |

Each invocation contains:

| Field | Type | Description |
|---|---|---|
| `invocation_id` | string | Unique turn identifier |
| `user_content` | string | What the user said |
| `final_response` | string or null | The agent's final response |
| `tool_calls` | array | Tools the agent called |
| `tool_responses` | array | Responses the agent received from tools |

### Output (`EvalResult`)

```json
{
  "score": 0.85,
  "status": null,
  "per_invocation_scores": [1.0, 0.7],
  "details": { "issues": ["inv-002: response too short"] }
}
```

| Field | Required | Description |
|---|---|---|
| `score` | yes | Overall score between 0.0 and 1.0 |
| `status` | no | `"PASSED"`, `"FAILED"`, or `"NOT_EVALUATED"`. If omitted, derived from score vs threshold. |
| `per_invocation_scores` | no | Per-turn scores (same order as input invocations) |
| `details` | no | Arbitrary metadata for debugging |

## Writing Graders in Other Languages

You don't need the Python SDK. Any program that reads JSON from stdin and writes JSON to stdout works.

### JavaScript / TypeScript

```javascript
// graders/tool_check.js
const input = JSON.parse(require("fs").readFileSync("/dev/stdin", "utf8"));

let score = 1.0;
for (const inv of input.invocations) {
  if (inv.tool_calls.length === 0) {
    score -= 0.5;
  }
}

console.log(JSON.stringify({
  score: Math.max(0, score),
  per_invocation_scores: [],
}));
```

```yaml
- name: tool_check
  type: code
  path: ./graders/tool_check.js
```

### Any language

Write a program that:

1. Reads all of stdin as a UTF-8 string
2. Parses it as JSON (matching the `EvalInput` schema)
3. Writes a JSON object to stdout (matching the `EvalResult` schema)
4. Exits with code 0 on success, non-zero on failure

The file extension determines which interpreter is used:

| Extension | Command |
|---|---|
| `.py` | `python <file>` |
| `.js`, `.ts` | `node <file>` |

## Discovering Graders

### List available graders

```bash
agentevals grader list                    # all sources
agentevals grader list --source builtin   # only ADK built-in metrics
agentevals grader list --source github    # only community graders
```

This shows graders from all registered sources: ADK built-in metrics and the community GitHub repository.

## Remote Graders

You can reference graders from the community repository directly in your eval config. They are downloaded and cached automatically on first use.

```yaml
metrics:
  - tool_trajectory_avg_score

  - name: response_quality
    type: remote
    source: github
    ref: graders/response_quality/response_quality.py
    threshold: 0.7
```

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | yes | | Unique name for the grader (used in output) |
| `type` | yes | | `remote` for graders fetched from a registry |
| `source` | no | `github` | Grader source (`github`, or custom) |
| `ref` | yes | | Path within the source (e.g. path in the GitHub repo) |
| `threshold` | no | `0.5` | Score at or above this value means PASSED |
| `timeout` | no | `30` | Subprocess timeout in seconds |
| `config` | no | `{}` | Arbitrary key-value pairs passed to the grader |
| `executor` | no | `local` | Execution environment (`local` or `docker` in the future) |

Remote graders are cached in `~/.cache/agentevals/graders/`. To force a re-download, delete the cached file.

### Configuring the GitHub source

By default, graders are fetched from the official community repository. Override with environment variables:

```bash
export AGENTEVALS_GRADER_REPO="your-org/your-graders-repo"
export AGENTEVALS_GRADER_BRANCH="main"
```

## Contributing Graders to the Community

1. Scaffold a new grader:

```bash
agentevals grader init my_grader
```

2. Implement your scoring logic and update the `grader.yaml` manifest with a description, tags, and your name.

3. Copy the `my_grader/` directory into the `graders/` folder of the community repository and open a PR.

The community repo uses per-grader manifests. A CI workflow compiles all `graders/*/grader.yaml` files into a single `index.yaml` on merge, which is what `agentevals grader list` fetches.

## Architecture

Custom graders use a layered architecture designed for extensibility.

```
┌─────────────────────────────────────────┐
│  Eval Config (YAML)                     │
│  type: code | remote                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  GraderResolver                         │
│  Downloads remote → local cache         │
│  (passthrough for type: code)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  CustomGraderEvaluator                  │
│  ADK Evaluator adapter                  │
│  Invocation ↔ EvalInput/EvalResult      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  GraderBackend (ABC) — executor factory │
│  async run(EvalInput) → EvalResult      │
├─────────────────────────────────────────┤
│  "local"  → SubprocessBackend           │
│  "docker" → DockerBackend (future)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Runtime registry                       │
│  PythonRuntime (.py)                    │
│  NodeRuntime (.js, .ts)                 │
└─────────────────────────────────────────┘
```

- **`GraderSource`** is the registry abstraction. Implementations (`BuiltinGraderSource`, `GitHubGraderSource`) list and fetch graders from different registries.
- **`GraderResolver`** downloads remote graders and converts `RemoteGraderDef` to `CodeGraderDef` with a local cached path.
- **`GraderBackend`** is the execution abstraction. The `executor` field in config selects which factory to use (`"local"` → `SubprocessBackend`). New executors (e.g. `DockerBackend`) register via `register_executor()`.
- **`SubprocessBackend`** runs a local file as a child process, piping JSON over stdin/stdout.
- **`Runtime`** is an internal detail of `SubprocessBackend` that maps file extensions to interpreter commands.
- **`CustomGraderEvaluator`** adapts any `GraderBackend` into ADK's `Evaluator` interface, handling the conversion between ADK's `Invocation` objects and the simpler `EvalInput`/`EvalResult` protocol.

### Adding a new language runtime

To support a new language (e.g., Go), add a `Runtime` subclass in `custom_evaluators.py`:

```python
class GoRuntime(Runtime):
    @property
    def extensions(self) -> tuple[str, ...]:
        return (".go",)

    def build_command(self, path: Path) -> list[str]:
        go = shutil.which("go")
        if not go:
            raise RuntimeError("Go not found on PATH")
        return [go, "run", str(path)]
```

Then register it:

```python
_RUNTIMES: list[Runtime] = [
    PythonRuntime(),
    NodeRuntime(),
    GoRuntime(),       # new
]
```

No other files need to change — the extension validator and evaluator pick it up automatically.

### Adding a new executor

To support a different execution environment (e.g., Docker), you need two things:

1. Implement the backend in `custom_evaluators.py`:

```python
class DockerBackend(GraderBackend):
    def __init__(self, path: Path, timeout: int = 30):
        self._path = path
        self._timeout = timeout

    async def run(self, eval_input: EvalInput, metric_name: str) -> EvalResult:
        # Build/run container, pipe JSON, return result
        ...
```

2. Register it:

```python
from agentevals.custom_evaluators import register_executor

register_executor("docker", lambda path, timeout: DockerBackend(path, timeout))
```

Users then set `executor: docker` in their config:

```yaml
metrics:
  - name: untrusted_grader
    type: code
    path: ./graders/untrusted.py
    executor: docker
```

### Adding a new grader source

To support a different grader registry (e.g., a custom API), implement `GraderSource`:

```python
from agentevals.grader.sources import GraderSource, GraderInfo, register_source

class MyRegistrySource(GraderSource):
    @property
    def source_name(self) -> str:
        return "my-registry"

    async def list_graders(self) -> list[GraderInfo]: ...
    async def fetch_grader(self, ref: str, dest: Path) -> Path: ...

register_source(MyRegistrySource())
```

Users can then reference graders from the new source:

```yaml
metrics:
  - name: my_grader
    type: remote
    source: my-registry
    ref: some/ref/path.py
```
