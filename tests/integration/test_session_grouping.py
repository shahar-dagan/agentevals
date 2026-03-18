"""Session grouping edge case tests.

Deterministic tests using crafted OTLP payloads via ASGI transport.
Each test gets a fresh StreamingTraceManager with fast timers (0.1s/0.5s).
"""

from __future__ import annotations

import asyncio

import pytest

from .conftest import (
    make_genai_log,
    make_genai_span,
    make_log_request,
    make_trace_request,
    send_logs,
    send_traces,
    wait_for_session_complete,
    get_sessions,
)

pytestmark = pytest.mark.integration


class TestBasicSessionCreation:
    async def test_single_trace_creates_session(self, trace_manager, otlp_client):
        body = make_trace_request(
            trace_id="t1",
            session_name="basic",
            spans=[make_genai_span(trace_id="t1", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "basic")

        session = trace_manager.sessions["basic"]
        assert session.is_complete
        assert session.source == "otlp"
        assert len(session.spans) == 1
        assert "t1" in session.trace_ids

    async def test_session_name_from_resource_attrs(self, trace_manager, otlp_client):
        body = make_trace_request(
            trace_id="t1",
            session_name="custom-name",
            spans=[make_genai_span(trace_id="t1", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "custom-name")

        assert "custom-name" in trace_manager.sessions

    async def test_fallback_session_name(self, trace_manager, otlp_client):
        """Without agentevals.session_name, falls back to otlp-{trace_id[:12]}."""
        body = make_trace_request(
            trace_id="abcdef123456789",
            session_name=None,
            spans=[make_genai_span(trace_id="abcdef123456789", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "otlp-abcdef123456")

        assert "otlp-abcdef123456" in trace_manager.sessions

    async def test_eval_set_id_propagated(self, trace_manager, otlp_client):
        body = make_trace_request(
            trace_id="t1",
            session_name="eval-test",
            eval_set_id="my-eval-set",
            spans=[make_genai_span(trace_id="t1", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "eval-test")

        assert trace_manager.sessions["eval-test"].eval_set_id == "my-eval-set"


class TestMultiTraceGrouping:
    async def test_multi_trace_same_session(self, trace_manager, otlp_client):
        """Multiple traces with same session_name → one session."""
        for i in range(3):
            body = make_trace_request(
                trace_id=f"trace-{i}",
                session_name="multi",
                spans=[make_genai_span(trace_id=f"trace-{i}")],
            )
            await send_traces(otlp_client, body)

        # Send a root span to trigger completion
        body = make_trace_request(
            trace_id="trace-root",
            session_name="multi",
            spans=[make_genai_span(trace_id="trace-root", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "multi")

        session = trace_manager.sessions["multi"]
        assert len(session.spans) == 4
        assert session.trace_ids == {"trace-0", "trace-1", "trace-2", "trace-root"}

    async def test_different_names_different_sessions(self, trace_manager, otlp_client):
        for name in ["session-a", "session-b"]:
            body = make_trace_request(
                trace_id=f"trace-{name}",
                session_name=name,
                spans=[make_genai_span(trace_id=f"trace-{name}", parent_span_id=None)],
            )
            await send_traces(otlp_client, body)

        await wait_for_session_complete(trace_manager, "session-a")
        await wait_for_session_complete(trace_manager, "session-b")

        assert "session-a" in trace_manager.sessions
        assert "session-b" in trace_manager.sessions
        assert trace_manager.sessions["session-a"].trace_ids != trace_manager.sessions["session-b"].trace_ids


class TestSessionCompletion:
    async def test_root_span_triggers_completion(self, trace_manager, otlp_client):
        body = make_trace_request(
            trace_id="t1",
            session_name="root-test",
            spans=[
                make_genai_span(trace_id="t1", span_id="child1", parent_span_id="root1"),
                make_genai_span(trace_id="t1", span_id="root1", parent_span_id=None),
            ],
        )
        await send_traces(otlp_client, body)

        # Should complete within grace period (0.1s) + tolerance
        await wait_for_session_complete(trace_manager, "root-test", timeout=1.0)

        session = trace_manager.sessions["root-test"]
        assert session.is_complete
        assert session.has_root_span
        assert len(session.spans) == 2

    async def test_no_root_span_idle_timeout(self, trace_manager, otlp_client):
        """Without root span, session completes via idle timeout (0.5s in tests)."""
        body = make_trace_request(
            trace_id="t1",
            session_name="idle-test",
            spans=[make_genai_span(trace_id="t1", parent_span_id="some-parent")],
        )
        await send_traces(otlp_client, body)

        session = trace_manager.sessions["idle-test"]
        assert not session.is_complete
        assert not session.has_root_span

        await wait_for_session_complete(trace_manager, "idle-test", timeout=2.0)
        assert session.is_complete


class TestSessionNameCollisions:
    async def test_sequential_runs_unique_ids(self, trace_manager, otlp_client):
        """Repeated runs with same name after completion → name-2, name-3."""
        # First run
        body = make_trace_request(
            trace_id="run1",
            session_name="repeated",
            spans=[make_genai_span(trace_id="run1", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "repeated")

        # Second run — same name but first is complete
        body = make_trace_request(
            trace_id="run2",
            session_name="repeated",
            spans=[make_genai_span(trace_id="run2", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "repeated-2")

        # Third run
        body = make_trace_request(
            trace_id="run3",
            session_name="repeated",
            spans=[make_genai_span(trace_id="run3", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "repeated-3")

        assert "repeated" in trace_manager.sessions
        assert "repeated-2" in trace_manager.sessions
        assert "repeated-3" in trace_manager.sessions
        assert trace_manager.sessions["repeated"].trace_ids == {"run1"}
        assert trace_manager.sessions["repeated-2"].trace_ids == {"run2"}
        assert trace_manager.sessions["repeated-3"].trace_ids == {"run3"}


class TestOrphanLogs:
    async def test_orphan_logs_before_spans(self, trace_manager, otlp_client):
        """Logs arriving before spans are buffered and replayed into session."""
        log_body = make_log_request(
            trace_id="orphan-trace",
            session_name="orphan-test",
            log_records=[
                make_genai_log("gen_ai.user.message", "Hello", trace_id="orphan-trace"),
            ],
        )
        await send_logs(otlp_client, log_body)

        assert len(trace_manager._orphan_logs) == 1

        trace_body = make_trace_request(
            trace_id="orphan-trace",
            session_name="orphan-test",
            spans=[make_genai_span(trace_id="orphan-trace", parent_span_id=None)],
        )
        await send_traces(otlp_client, trace_body)
        await wait_for_session_complete(trace_manager, "orphan-test")

        session = trace_manager.sessions["orphan-test"]
        assert len(session.logs) >= 1
        assert len(trace_manager._orphan_logs) == 0

    async def test_orphan_logs_matched_by_session_name(self, trace_manager, otlp_client):
        """Orphan log with matching session_name but different trace_id → replayed."""
        log_body = make_log_request(
            trace_id="log-trace",
            session_name="name-match",
            log_records=[
                make_genai_log("gen_ai.user.message", "Hi", trace_id="log-trace"),
            ],
        )
        await send_logs(otlp_client, log_body)

        # Spans arrive with a different trace_id but same session_name
        trace_body = make_trace_request(
            trace_id="span-trace",
            session_name="name-match",
            spans=[make_genai_span(trace_id="span-trace", parent_span_id=None)],
        )
        await send_traces(otlp_client, trace_body)
        await wait_for_session_complete(trace_manager, "name-match")

        session = trace_manager.sessions["name-match"]
        assert "log-trace" in session.trace_ids
        assert "span-trace" in session.trace_ids
        assert len(session.logs) >= 1


class TestLateLogs:
    async def test_late_logs_after_completion(self, trace_manager, otlp_client):
        """Logs arriving after session completion trigger re-extraction."""
        trace_body = make_trace_request(
            trace_id="late-trace",
            session_name="late-test",
            spans=[make_genai_span(trace_id="late-trace", parent_span_id=None)],
        )
        await send_traces(otlp_client, trace_body)
        await wait_for_session_complete(trace_manager, "late-test")

        session = trace_manager.sessions["late-test"]
        assert session.is_complete
        assert len(session.logs) == 0

        log_body = make_log_request(
            trace_id="late-trace",
            session_name="late-test",
            log_records=[
                make_genai_log("gen_ai.user.message", "Late hello", trace_id="late-trace"),
            ],
        )
        await send_logs(otlp_client, log_body)

        # Wait for re-extraction debounce (0.1s in tests)
        await asyncio.sleep(0.3)

        assert len(session.logs) == 1


class TestAPIVisibility:
    async def test_sessions_visible_via_api(
        self, trace_manager, otlp_client, api_client
    ):
        body = make_trace_request(
            trace_id="vis-trace",
            session_name="visible",
            spans=[make_genai_span(trace_id="vis-trace", parent_span_id=None)],
        )
        await send_traces(otlp_client, body)
        await wait_for_session_complete(trace_manager, "visible")

        sessions = await get_sessions(api_client)
        session_ids = [s["sessionId"] for s in sessions]
        assert "visible" in session_ids


class TestSpanLimits:
    async def test_span_limit_enforcement(self, trace_manager, otlp_client):
        """Session rejects spans beyond MAX_SPANS_PER_SESSION."""
        from agentevals.streaming.session import MAX_SPANS_PER_SESSION

        # Pre-fill the session with spans up to the limit
        body = make_trace_request(
            trace_id="limit-trace",
            session_name="limit-test",
            spans=[make_genai_span(trace_id="limit-trace")],
        )
        await send_traces(otlp_client, body)

        session = trace_manager.sessions["limit-test"]
        session.spans.extend([{}] * (MAX_SPANS_PER_SESSION - 1))
        assert not session.can_accept_span()

        # This span should be rejected
        body2 = make_trace_request(
            trace_id="limit-trace",
            session_name="limit-test",
            spans=[make_genai_span(trace_id="limit-trace")],
        )
        await send_traces(otlp_client, body2)

        assert len(session.spans) == MAX_SPANS_PER_SESSION
