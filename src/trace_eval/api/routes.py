"""API routes for trace-eval."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import EvalRunConfig
from ..runner import RunResult, load_eval_set, run_evaluation

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@router.get("/metrics")
async def list_metrics():
    """List available metrics with metadata.

    Dynamically loads metrics from ADK registry and adds trace-eval metadata.
    """
    from ..runner import _METRICS_NEEDING_EXPECTED, _METRICS_NEEDING_LLM

    _METRICS_NEEDING_GCP = {
        "response_evaluation_score",
        "safety_v1",
    }

    _METRICS_NEEDING_RUBRICS = {
        "rubric_based_final_response_quality_v1",
        "rubric_based_tool_use_quality_v1",
    }

    _METRIC_CATEGORIES = {
        "tool_trajectory_avg_score": "trajectory",
        "response_match_score": "response",
        "response_evaluation_score": "response",
        "final_response_match_v2": "response",
        "rubric_based_final_response_quality_v1": "quality",
        "rubric_based_tool_use_quality_v1": "quality",
        "hallucinations_v1": "safety",
        "safety_v1": "safety",
    }

    try:
        # Try to load from ADK registry (like CLI does)
        from google.adk.evaluation.metric_evaluator_registry import (
            DEFAULT_METRIC_EVALUATOR_REGISTRY,
        )

        registry_metrics = DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()

        # Filter out per_turn_user_simulator_quality_v1 (not applicable to trace eval)
        metrics = []
        for m in registry_metrics:
            if m.metric_name == "per_turn_user_simulator_quality_v1":
                continue

            metrics.append({
                "name": m.metric_name,
                "category": _METRIC_CATEGORIES.get(m.metric_name, "other"),
                "requiresEvalSet": m.metric_name in _METRICS_NEEDING_EXPECTED,
                "requiresLLM": m.metric_name in _METRICS_NEEDING_LLM,
                "requiresGCP": m.metric_name in _METRICS_NEEDING_GCP,
                "requiresRubrics": m.metric_name in _METRICS_NEEDING_RUBRICS,
                "description": m.description or "No description available",
                "working": m.metric_name not in _METRICS_NEEDING_RUBRICS,
            })

        return metrics

    except ImportError:
        return [
            {
                "name": "tool_trajectory_avg_score",
                "category": "trajectory",
                "requiresEvalSet": True,
                "requiresLLM": False,
                "requiresGCP": False,
                "requiresRubrics": False,
                "working": True,
                "description": "Compare tool call sequences against expected trajectory",
            },
            {
                "name": "response_match_score",
                "category": "response",
                "requiresEvalSet": True,
                "requiresLLM": False,
                "requiresGCP": False,
                "requiresRubrics": False,
                "working": True,
                "description": "Text similarity between actual and expected responses using ROUGE-1",
            },
            {
                "name": "response_evaluation_score",
                "category": "response",
                "requiresEvalSet": True,
                "requiresLLM": False,
                "requiresGCP": True,
                "requiresRubrics": False,
                "working": True,
                "description": "Semantic evaluation of response quality using Vertex AI",
            },
            {
                "name": "final_response_match_v2",
                "category": "response",
                "requiresEvalSet": True,
                "requiresLLM": True,
                "requiresGCP": False,
                "requiresRubrics": False,
                "working": True,
                "description": "LLM-based comparison of final responses",
            },
            {
                "name": "hallucinations_v1",
                "category": "safety",
                "requiresEvalSet": False,
                "requiresLLM": True,
                "requiresGCP": False,
                "requiresRubrics": False,
                "working": True,
                "description": "Detect hallucinated information in responses",
            },
            {
                "name": "safety_v1",
                "category": "safety",
                "requiresEvalSet": False,
                "requiresLLM": False,
                "requiresGCP": True,
                "requiresRubrics": False,
                "working": True,
                "description": "Safety and security assessment using Vertex AI",
            },
            {
                "name": "rubric_based_final_response_quality_v1",
                "category": "quality",
                "requiresEvalSet": False,
                "requiresLLM": True,
                "requiresGCP": False,
                "requiresRubrics": True,
                "working": False,
                "description": "Rubric-based quality assessment of responses (requires rubrics config)",
            },
            {
                "name": "rubric_based_tool_use_quality_v1",
                "category": "quality",
                "requiresEvalSet": False,
                "requiresLLM": True,
                "requiresGCP": False,
                "requiresRubrics": True,
                "working": False,
                "description": "Rubric-based assessment of tool usage quality (requires rubrics config)",
            },
        ]


@router.post("/validate/eval-set")
async def validate_eval_set(
    eval_set_file: UploadFile = File(...),
):
    """Validate an eval set file structure."""
    temp_dir = tempfile.mkdtemp()
    try:
        eval_set_path = os.path.join(temp_dir, eval_set_file.filename or "eval_set.json")
        with open(eval_set_path, "wb") as f:
            content = await eval_set_file.read()
            f.write(content)

        try:
            eval_set = load_eval_set(eval_set_path)
            return {
                "valid": True,
                "eval_set_id": eval_set.eval_set_id,
                "num_cases": len(eval_set.eval_cases),
                "errors": [],
            }
        except Exception as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
            }

    finally:
        shutil.rmtree(temp_dir)


@router.post("/evaluate", response_model=RunResult)
async def evaluate_traces(
    trace_files: list[UploadFile] = File(...),
    config: str = Form(...),
    eval_set_file: Optional[UploadFile] = File(None),
):
    """
    Evaluate agent traces using specified metrics.

    Args:
        trace_files: List of Jaeger JSON trace files
        config: JSON string with evaluation configuration
        eval_set_file: Optional golden eval set file

    Returns:
        RunResult with trace results and any errors
    """
    temp_dir = tempfile.mkdtemp()
    try:
        try:
            config_dict = json.loads(config)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}")

        trace_paths = []
        for trace_file in trace_files:
            if not trace_file.filename:
                continue

            if not trace_file.filename.endswith(".json"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension for {trace_file.filename}. Only .json files are allowed.",
                )

            trace_path = os.path.join(temp_dir, trace_file.filename)
            with open(trace_path, "wb") as f:
                content = await trace_file.read()

                if len(content) > 10 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {trace_file.filename} exceeds 10MB limit",
                    )

                f.write(content)
            trace_paths.append(trace_path)

        if not trace_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid trace files provided",
            )

        eval_set_path = None
        if eval_set_file and eval_set_file.filename:
            if not eval_set_file.filename.endswith(".json"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension for eval set. Only .json files are allowed.",
                )

            eval_set_path = os.path.join(temp_dir, eval_set_file.filename)
            with open(eval_set_path, "wb") as f:
                content = await eval_set_file.read()
                if len(content) > 10 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail="Eval set file exceeds 10MB limit",
                    )
                f.write(content)

        metrics = config_dict.get("metrics", ["tool_trajectory_avg_score"])
        if not metrics or not isinstance(metrics, list):
            raise HTTPException(
                status_code=400,
                detail="Config must include 'metrics' as a non-empty array",
            )

        threshold = config_dict.get("threshold")
        if threshold is not None and (threshold < 0 or threshold > 1):
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0 and 1",
            )

        eval_config = EvalRunConfig(
            trace_files=trace_paths,
            eval_set_file=eval_set_path,
            metrics=metrics,
            trace_format=config_dict.get("trace_format", "jaeger-json"),
            judge_model=config_dict.get("judgeModel"),  # camelCase from UI
            threshold=threshold,
        )

        logger.info(f"Evaluating {len(trace_paths)} trace file(s) with metrics: {metrics}")
        result = await run_evaluation(eval_config)

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")

    finally:
        shutil.rmtree(temp_dir)
