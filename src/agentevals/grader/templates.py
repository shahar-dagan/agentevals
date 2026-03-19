"""Grader scaffolding templates and the scaffold_grader function."""

from __future__ import annotations

from pathlib import Path
from string import Template

import yaml

PYTHON_TEMPLATE = Template('''\
"""Custom grader: ${name}

Usage in eval_config.yaml:

    metrics:
      - name: ${name}
        type: code
        path: ./${name}/${name}.py
        threshold: 0.5
"""

from agentevals_grader_sdk import grader, EvalInput, EvalResult


@grader
def ${name}(input: EvalInput) -> EvalResult:
    scores: list[float] = []

    for inv in input.invocations:
        score = 1.0

        if not inv.final_response:
            score = 0.0
            scores.append(score)
            continue

        # TODO: implement your scoring logic here

        scores.append(max(0.0, score))

    overall = sum(scores) / len(scores) if scores else 0.0
    return EvalResult(
        score=overall,
        per_invocation_scores=scores,
    )
''')


JAVASCRIPT_TEMPLATE = Template("""\
/**
 * Custom grader: ${name}
 *
 * Usage in eval_config.yaml:
 *
 *     metrics:
 *       - name: ${name}
 *         type: code
 *         path: ./${name}/${name}.js
 *         threshold: 0.5
 */

const input = JSON.parse(require("fs").readFileSync("/dev/stdin", "utf8"));

const scores = [];

for (const inv of input.invocations) {
  let score = 1.0;

  if (!inv.final_response) {
    scores.push(0.0);
    continue;
  }

  // TODO: implement your scoring logic here

  scores.push(Math.max(0.0, score));
}

const overall = scores.length > 0
  ? scores.reduce((a, b) => a + b, 0) / scores.length
  : 0.0;

console.log(JSON.stringify({
  score: overall,
  per_invocation_scores: scores,
}));
""")


TYPESCRIPT_TEMPLATE = Template("""\
/**
 * Custom grader: ${name}
 *
 * Usage in eval_config.yaml:
 *
 *     metrics:
 *       - name: ${name}
 *         type: code
 *         path: ./${name}/${name}.ts
 *         threshold: 0.5
 */

import * as fs from "fs";

interface Invocation {
  invocation_id: string;
  user_content: string;
  final_response: string | null;
  tool_calls: { name: string; args: Record<string, unknown> }[];
  tool_responses: { name: string; output: string }[];
}

interface EvalInput {
  metric_name: string;
  threshold: number;
  config: Record<string, unknown>;
  invocations: Invocation[];
  expected_invocations: Invocation[] | null;
}

const input: EvalInput = JSON.parse(fs.readFileSync("/dev/stdin", "utf8"));

const scores: number[] = [];

for (const inv of input.invocations) {
  let score = 1.0;

  if (!inv.final_response) {
    scores.push(0.0);
    continue;
  }

  // TODO: implement your scoring logic here

  scores.push(Math.max(0.0, score));
}

const overall = scores.length > 0
  ? scores.reduce((a, b) => a + b, 0) / scores.length
  : 0.0;

console.log(JSON.stringify({
  score: overall,
  per_invocation_scores: scores,
}));
""")


_EXTENSION_TO_TEMPLATE: dict[str, Template] = {
    ".py": PYTHON_TEMPLATE,
    ".js": JAVASCRIPT_TEMPLATE,
    ".ts": TYPESCRIPT_TEMPLATE,
}

_RUNTIME_ALIAS_TO_EXT: dict[str, str] = {
    "py": ".py",
    "python": ".py",
    "js": ".js",
    "javascript": ".js",
    "ts": ".ts",
    "typescript": ".ts",
}

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
}


def scaffold_grader(
    name: str,
    output_dir: Path | None = None,
    runtime: str | None = None,
) -> Path:
    """Create a new grader directory with code file and grader.yaml manifest.

    Returns the path to the created directory.
    """
    output_dir = output_dir or Path.cwd()

    raw_path = Path(name)
    suffix = raw_path.suffix.lower()
    grader_name = raw_path.stem

    if suffix and suffix in _EXTENSION_TO_TEMPLATE:
        ext = suffix
    elif runtime:
        ext = _RUNTIME_ALIAS_TO_EXT.get(runtime.lower())
        if ext is None:
            raise ValueError(f"Unknown runtime '{runtime}'. Supported: {sorted(_RUNTIME_ALIAS_TO_EXT.keys())}")
    else:
        ext = ".py"

    template = _EXTENSION_TO_TEMPLATE[ext]
    language = _EXT_TO_LANGUAGE[ext]

    grader_dir = output_dir / grader_name
    grader_dir.mkdir(parents=True, exist_ok=True)

    code_file = grader_dir / f"{grader_name}{ext}"
    code_file.write_text(template.substitute(name=grader_name), encoding="utf-8")

    manifest = {
        "name": grader_name,
        "description": f"TODO: describe what {grader_name} evaluates",
        "language": language,
        "entrypoint": f"{grader_name}{ext}",
        "tags": [],
        "author": "",
    }
    manifest_file = grader_dir / "grader.yaml"
    manifest_file.write_text(yaml.dump(manifest, sort_keys=False), encoding="utf-8")

    return grader_dir
