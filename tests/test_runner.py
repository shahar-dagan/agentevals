import os

import pytest

from agentevals.config import EvalRunConfig
from agentevals.runner import run_evaluation, load_eval_set


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
HELM_TRACE = os.path.join(SAMPLES_DIR, "helm.json")
EVAL_SET = os.path.join(SAMPLES_DIR, "eval_set_helm.json")


@pytest.mark.skipif(
    not os.path.exists(HELM_TRACE) or not os.path.exists(EVAL_SET),
    reason="Sample files not available",
)
class TestRunner:
    def test_trajectory_eval_pass(self):
        """Helm trace should score 1.0 against its golden eval set."""
        config = EvalRunConfig(
            trace_files=[HELM_TRACE],
            eval_set_file=EVAL_SET,
            metrics=["tool_trajectory_avg_score"],
        )
        result = run_evaluation(config)

        assert len(result.errors) == 0
        assert len(result.trace_results) == 1

        tr = result.trace_results[0]
        assert tr.num_invocations == 1
        assert len(tr.metric_results) == 1

        mr = tr.metric_results[0]
        assert mr.metric_name == "tool_trajectory_avg_score"
        assert mr.score == 1.0
        assert mr.eval_status == "PASSED"
        assert mr.error is None

    def test_missing_eval_set_error(self):
        """Trajectory metric without eval set should report a clear error."""
        config = EvalRunConfig(
            trace_files=[HELM_TRACE],
            metrics=["tool_trajectory_avg_score"],
        )
        result = run_evaluation(config)

        mr = result.trace_results[0].metric_results[0]
        assert mr.error is not None
        assert "requires expected invocations" in mr.error

    def test_bad_trace_file(self):
        config = EvalRunConfig(
            trace_files=["/nonexistent/file.json"],
            metrics=["tool_trajectory_avg_score"],
        )
        result = run_evaluation(config)
        assert len(result.errors) >= 1

    def test_load_eval_set(self):
        eval_set = load_eval_set(EVAL_SET)
        assert eval_set.eval_set_id == "helm_eval_set"
        assert len(eval_set.eval_cases) == 1
        case = eval_set.eval_cases[0]
        assert case.eval_id == "helm_list_releases"
        assert case.conversation is not None
        assert len(case.conversation) == 1
        inv = case.conversation[0]
        assert inv.user_content.parts[0].text == "list all Helm releases"

    def test_multiple_metrics(self):
        config = EvalRunConfig(
            trace_files=[HELM_TRACE],
            eval_set_file=EVAL_SET,
            metrics=["tool_trajectory_avg_score", "tool_trajectory_avg_score"],
        )
        result = run_evaluation(config)

        tr = result.trace_results[0]
        assert len(tr.metric_results) == 2

    def test_json_output_format(self):
        from agentevals.output import format_results

        config = EvalRunConfig(
            trace_files=[HELM_TRACE],
            eval_set_file=EVAL_SET,
            metrics=["tool_trajectory_avg_score"],
        )
        result = run_evaluation(config)
        output = format_results(result, fmt="json")

        import json

        data = json.loads(output)
        assert "traces" in data
        assert len(data["traces"]) == 1
        assert data["traces"][0]["metrics"][0]["score"] == 1.0
