"""CLI entry point for trace-eval.

Usage::

    trace-eval run samples/helm.json --eval-set samples/eval_set_helm.json
    trace-eval run samples/helm.json -m tool_trajectory_avg_score -m response_match_score
    trace-eval run samples/helm.json --eval-set samples/eval_set_helm.json --output json
    trace-eval list-metrics
"""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from .config import EvalRunConfig
from .output import format_results
from .runner import run_evaluation


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for INFO, -vv for DEBUG).",
)
def main(verbose: int) -> None:
    """trace-eval: Evaluate agent traces using ADK's scoring framework."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


@main.command()
@click.argument("trace_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--eval-set",
    "-e",
    type=click.Path(exists=True),
    default=None,
    help="Path to a golden eval set JSON file (ADK EvalSet format).",
)
@click.option(
    "--metric",
    "-m",
    multiple=True,
    default=["tool_trajectory_avg_score"],
    help="Metric(s) to evaluate. Can be specified multiple times.",
)
@click.option(
    "--format",
    "-f",
    "trace_format",
    default="jaeger-json",
    help="Trace file format.",
)
@click.option(
    "--judge-model",
    "-j",
    default=None,
    help="LLM model for judge-based metrics (default: gemini-2.5-flash).",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=None,
    help="Score threshold for pass/fail.",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["table", "json", "summary"]),
    default="table",
    help="Output format.",
)
def run(
    trace_files: tuple[str, ...],
    eval_set: str | None,
    metric: tuple[str, ...],
    trace_format: str,
    judge_model: str | None,
    threshold: float | None,
    output: str,
) -> None:
    """Evaluate trace file(s) against specified metrics."""
    config = EvalRunConfig(
        trace_files=list(trace_files),
        eval_set_file=eval_set,
        metrics=list(metric),
        trace_format=trace_format,
        judge_model=judge_model,
        threshold=threshold,
        output_format=output,
    )

    result = asyncio.run(run_evaluation(config))
    formatted = format_results(result, fmt=output)
    click.echo(formatted)

    has_failure = any(
        mr.eval_status == "FAILED" or mr.error
        for tr in result.trace_results
        for mr in tr.metric_results
    )
    if has_failure or result.errors:
        sys.exit(1)


@main.command("list-metrics")
def list_metrics() -> None:
    """List all available evaluation metrics."""
    try:
        from google.adk.evaluation.metric_evaluator_registry import (
            DEFAULT_METRIC_EVALUATOR_REGISTRY,
        )

        metrics = DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()
        click.echo("Available metrics:\n")
        for m in metrics:
            desc = m.description or "No description"
            click.echo(f"  {m.metric_name}")
            click.echo(f"    {desc}")
            if m.metric_value_info and m.metric_value_info.interval:
                iv = m.metric_value_info.interval
                lo = f"{'(' if iv.open_at_min else '['}{iv.min_value}"
                hi = f"{iv.max_value}{')' if iv.open_at_max else ']'}"
                click.echo(f"    Value range: {lo}, {hi}")
            click.echo()
    except ImportError as exc:
        # Full registry needs rouge_score/numpy — show what we can
        click.echo(
            f"Could not load full metric registry ({exc}).\n"
            "Some eval dependencies may be missing. Install with:\n"
            '  pip install "google-adk[eval]"\n'
        )
        click.echo("Known built-in metrics:\n")
        from google.adk.evaluation.eval_metrics import PrebuiltMetrics

        for pm in PrebuiltMetrics:
            click.echo(f"  {pm.value}")
        click.echo()


if __name__ == "__main__":
    main()
