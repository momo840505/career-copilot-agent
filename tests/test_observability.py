"""Pure-logic tests for career_copilot/observability.py -- no API key, no
LLM, no FastAPI TestClient needed (that part is covered separately in test_api.py's
GET /metrics tests). Everything here is stdlib (logging, json, threading), so these
run anywhere this package installs.
"""
from __future__ import annotations

import json
import logging

from career_copilot.observability import Metrics, _JsonFormatter, configure_logging

# --- _JsonFormatter ------------------------------------------------------------------


def _make_record(msg: str, *, extra: dict | None = None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_json_with_standard_fields():
    record = _make_record("hello world")
    line = _JsonFormatter().format(record)
    parsed = json.loads(line)  # raises if not valid JSON -- the whole point of the test
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert "timestamp" in parsed


def test_json_formatter_includes_extra_fields_without_duplicating_reserved_ones():
    record = _make_record("gap analysis attempt", extra={"node": "gap_analysis", "attempt": 2, "stuck": True})
    parsed = json.loads(_JsonFormatter().format(record))
    assert parsed["node"] == "gap_analysis"
    assert parsed["attempt"] == 2
    assert parsed["stuck"] is True
    # Reserved LogRecord attributes (e.g. "levelname", "msg", "args") must not leak
    # through as raw extra keys -- only the friendly "level"/"message" pair should
    # appear, or every log line would be cluttered with logging-internal noise.
    assert "levelname" not in parsed
    assert "msg" not in parsed
    assert "args" not in parsed


def test_configure_logging_is_idempotent(monkeypatch):
    # A second call must not attach a second handler -- otherwise every log line
    # would be duplicated (this exact bug would show up as doubled log output the
    # moment any module imports and configures logging more than once, e.g. once at
    # api/app.py's import time and again if a test or script called it directly).
    import career_copilot.observability as obs

    monkeypatch.setattr(obs, "_configured", False)
    configure_logging()
    root = logging.getLogger()
    handler_count_after_first = len(root.handlers)
    configure_logging()
    assert len(root.handlers) == handler_count_after_first


# --- Metrics ---------------------------------------------------------------------


def test_snapshot_of_a_fresh_registry_is_empty():
    m = Metrics()
    snap = m.snapshot()
    assert snap["routes"] == {}
    assert snap["llm_nodes"] == {}
    assert snap["uptime_seconds"] >= 0


def test_record_request_aggregates_count_status_and_average_latency():
    m = Metrics()
    m.record_request("/gap-analysis", 200, 100.0)
    m.record_request("/gap-analysis", 200, 200.0)
    m.record_request("/gap-analysis", 502, 50.0)

    route = m.snapshot()["routes"]["/gap-analysis"]
    assert route["count"] == 3
    assert route["status_counts"] == {"200": 2, "502": 1}
    assert route["avg_duration_ms"] == 116.7  # (100+200+50)/3, rounded to 1 decimal


def test_record_llm_call_tracks_retries_stuck_and_failures_per_node():
    m = Metrics()
    # First call: succeeded on the 1st attempt -- 0 retries.
    m.record_llm_call("gap_analysis", attempts=1, stuck_occurred=False, succeeded=True, duration_ms=500.0)
    # Second call: needed 3 attempts (2 retries) and hit the STUCK path, but still
    # eventually succeeded.
    m.record_llm_call("gap_analysis", attempts=3, stuck_occurred=True, succeeded=True, duration_ms=1500.0)
    # Third call: exhausted all 5 attempts (4 retries) and gave up.
    m.record_llm_call("gap_analysis", attempts=5, stuck_occurred=True, succeeded=False, duration_ms=3000.0)

    node = m.snapshot()["llm_nodes"]["gap_analysis"]
    assert node["calls"] == 3
    assert node["retries"] == 0 + 2 + 4
    assert node["stuck_retries"] == 2
    assert node["failures"] == 1


def test_different_nodes_and_routes_are_tracked_independently():
    m = Metrics()
    m.record_request("/gap-analysis", 200, 100.0)
    m.record_request("/draft", 200, 300.0)
    m.record_llm_call("parse_jd", attempts=1, stuck_occurred=False, succeeded=True, duration_ms=100.0)
    m.record_llm_call("critic", attempts=1, stuck_occurred=False, succeeded=True, duration_ms=200.0)

    snap = m.snapshot()
    assert set(snap["routes"]) == {"/gap-analysis", "/draft"}
    assert set(snap["llm_nodes"]) == {"parse_jd", "critic"}
    assert snap["routes"]["/gap-analysis"]["count"] == 1
    assert snap["routes"]["/draft"]["count"] == 1
