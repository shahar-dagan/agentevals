"""Configuration for agentevals runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator


class BuiltinMetricDef(BaseModel):
    """A built-in ADK metric, optionally with threshold/judge overrides."""

    name: str
    type: Literal["builtin"] = "builtin"
    threshold: float | None = None
    judge_model: str | None = None


class BaseGraderDef(BaseModel):
    """Shared fields for all executable grader definitions."""

    name: str
    threshold: float = 0.5
    timeout: int = Field(default=30, description="Subprocess timeout in seconds.")
    config: dict[str, Any] = Field(default_factory=dict)
    executor: str = Field(default="local", description="Execution environment: 'local' or 'docker' (future).")


class CodeGraderDef(BaseGraderDef):
    """A grader implemented as an external code file (Python, JavaScript, etc.)."""

    type: Literal["code"] = "code"
    path: str = Field(description="Path to the grader file (.py, .js, .ts, etc.).")

    @field_validator("path")
    @classmethod
    def _validate_extension(cls, v: str) -> str:
        from .custom_evaluators import supported_extensions

        suffix = Path(v).suffix.lower()
        allowed = supported_extensions()
        if suffix not in allowed:
            raise ValueError(f"Unsupported grader file extension '{suffix}'. Supported: {sorted(allowed)}")
        return v


class RemoteGraderDef(BaseGraderDef):
    """A grader fetched from a remote source (GitHub, registry, etc.)."""

    type: Literal["remote"] = "remote"
    source: str = Field(default="github", description="Grader source (e.g. 'github').")
    ref: str = Field(description="Source-specific reference (e.g. path within the repo).")


CustomGraderDef = Annotated[
    BuiltinMetricDef | CodeGraderDef | RemoteGraderDef,
    Field(discriminator="type"),
]


class EvalRunConfig(BaseModel):
    trace_files: list[str] = Field(description="Paths to trace files (Jaeger JSON or OTLP JSON).")

    eval_set_file: str | None = Field(
        default=None,
        description="Path to a golden eval set JSON file (ADK EvalSet format).",
    )

    metrics: list[str] = Field(
        default_factory=lambda: ["tool_trajectory_avg_score"],
        description="List of built-in metric names to evaluate.",
    )

    custom_graders: list[CustomGraderDef] = Field(
        default_factory=list,
        description="Custom grader definitions.",
    )

    trace_format: str = Field(
        default="jaeger-json",
        description="Format of the trace files (jaeger-json or otlp-json).",
    )

    judge_model: str | None = Field(
        default=None,
        description="LLM model for judge-based metrics.",
    )

    threshold: float | None = Field(
        default=None,
        description="Score threshold for pass/fail.",
    )

    output_format: str = Field(
        default="table",
        description="Output format: 'table', 'json', or 'summary'.",
    )

    max_concurrent_traces: int = Field(
        default=10,
        description="Maximum number of traces to evaluate concurrently.",
    )

    max_concurrent_evals: int = Field(
        default=5,
        description="Maximum number of concurrent metric evaluations (LLM API calls).",
    )
