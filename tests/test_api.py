"""Pure-logic tests for the FastAPI service -- no API key needed.

Every route handler in career_copilot.api.app calls the pipeline functions by name
imported into that module's namespace (run_parse_jd, run_retrieve_evidence,
run_gap_analysis, run_pipeline) and the db functions the same way (insert_history,
list_history, get_history), so monkeypatching those names there — the same pattern as
swapping a real LLM for a fake one in test_invoke_structured_retry.py — exercises the
real request/response wiring (validation, status codes, field mapping, access-code
gating) without ever calling OpenAI or touching a real SQLite file. Uses FastAPI's
TestClient (httpx under the hood, already pulled in transitively via
langchain-openai -> openai's own httpx dependency).
"""
from __future__ import annotations

import dataclasses

from fastapi.testclient import TestClient

from career_copilot.api import app as app_module
from career_copilot.api.db import HistoryRecord
from career_copilot.config import get_settings as real_get_settings
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.graph.structured import StructuredOutputError
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport
from career_copilot.schemas.jd import JDRequirements

client = TestClient(app_module.app)

# get_client_id (api/auth.py) requires this on every route except /health and
# /auth/verify, regardless of whether an access code is configured -- send it on
# every call below so tests exercise one thing at a time.
CLIENT_HEADERS = {"X-Client-Id": "test-client"}


def _jd() -> JDRequirements:
    return JDRequirements(
        job_title="Data Analyst",
        company="Acme",
        seniority="mid",
        must_have_skills=["SQL", "Python"],
        nice_to_have_skills=["Tableau"],
        keywords=["SQL", "Python", "Tableau"],
        summary="Analyze data and build dashboards.",
    )


def _bundle() -> EvidenceBundle:
    chunk = RetrievedChunk(
        chunk_id="skills::chunk0", doc_id="doc", doc_title="Resume", text="Used SQL daily.", distance=0.1
    )
    return EvidenceBundle(requirement="SQL", chunks=[chunk])


def _gap_report() -> GapReport:
    return GapReport(
        matched=[GapItem(requirement="SQL", evidence_chunk_ids=["skills::chunk0"], note="Direct match.")],
        partial=[],
        missing=[GapItem(requirement="Tableau", evidence_chunk_ids=[], note="No evidence found.")],
        suggested_talking_points=[],
        overall_fit_summary="Strong fit on core skills.",
    )


def _draft() -> CoverLetterDraft:
    return CoverLetterDraft(
        greeting="Dear Hiring Manager,",
        body="I have used SQL extensively.",
        closing="Sincerely,",
        claims=[Claim(text="I have used SQL extensively.", evidence_chunk_ids=["skills::chunk0"])],
    )


def _noop_insert_history(*args, **kwargs):
    """Stand-in for db.insert_history in tests that don't care about persistence --
    avoids ever touching a real SQLite file just because a route succeeded."""
    return


class _FakeInterrupt:
    """Stands in for langgraph.types.Interrupt: build_graph.py's human_review node
    calls interrupt(payload), and langgraph surfaces it back to invoke()'s caller as
    result["__interrupt__"][0].value -- only `.value` is ever read by api/app.py."""

    def __init__(self, value):
        self.value = value


class _FakeStateSnapshot:
    """Stands in for the object agent_graph.get_state(config) returns -- only
    `.values` (truthy iff a checkpoint exists for this thread_id) is read."""

    def __init__(self, values):
        self.values = values


def _pending_graph_result(*, critic_passed=True, revision_count=0, jd_text="Data Analyst role..."):
    """A fake agent_graph.invoke(...) return value shaped like what build_graph.py
    produces the moment human_review's interrupt() fires -- i.e. what every /draft
    call and every "revise" decision resumes into."""
    return {
        "jd_text": jd_text,
        "client_id": "test-client",
        "jd": _jd(),
        "evidence_bundles": [_bundle()],
        "gap_report": _gap_report(),
        "draft": _draft(),
        "critic_verdict": CriticVerdict(passed=critic_passed),
        "revision_count": revision_count,
        "__interrupt__": [
            _FakeInterrupt(
                {
                    "draft": _draft().model_dump(),
                    "critic_passed": critic_passed,
                    "critic_issues": [] if critic_passed else ["too generic"],
                    "gap_summary": _gap_report().overall_fit_summary,
                }
            )
        ],
    }


def _approved_graph_result(*, revision_count=0, jd_text="Data Analyst role..."):
    """A fake agent_graph.invoke(...) return value shaped like what build_graph.py
    produces once a human approves and the graph reaches END (no interrupt key)."""
    return {
        "jd_text": jd_text,
        "client_id": "test-client",
        "jd": _jd(),
        "evidence_bundles": [_bundle()],
        "gap_report": _gap_report(),
        "draft": _draft(),
        "critic_verdict": CriticVerdict(passed=True),
        "revision_count": revision_count,
        "human_decision": {"action": "approve"},
    }


def test_health_reports_ok_and_key_configured_flag_and_needs_no_client_id():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- GET /metrics ---


def test_metrics_returns_200_with_expected_shape_and_needs_no_client_id():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "uptime_seconds" in body
    assert "routes" in body
    assert "llm_nodes" in body


def test_metrics_stays_open_even_when_an_access_code_is_configured():
    # Same reasoning as /health: it's an ops endpoint (aggregate counts/latencies
    # only, nothing client-specific), not a data endpoint, so it must not join the
    # require_access_code gate the way /gap-analysis, /draft, and /history do.
    _with_access_code("secret123")
    try:
        response = client.get("/metrics")
        assert response.status_code == 200
    finally:
        _clear_access_code_override()


def test_metrics_reflects_requests_recorded_by_the_logging_middleware():
    before = client.get("/metrics").json()["routes"].get("/health", {}).get("count", 0)
    client.get("/health")
    after = client.get("/metrics").json()["routes"].get("/health", {}).get("count", 0)
    # +1 for the /health call itself; the two /metrics calls bracketing it land in
    # their own "/metrics" bucket, not "/health"'s, so they don't also bump this count.
    assert after == before + 1


def test_gap_analysis_maps_pipeline_result_to_response_fields(monkeypatch):
    monkeypatch.setattr(app_module, "run_parse_jd", lambda jd_text, settings: _jd())
    monkeypatch.setattr(app_module, "run_retrieve_evidence", lambda jd, settings: [_bundle()])
    monkeypatch.setattr(app_module, "run_gap_analysis", lambda jd, bundles, settings: _gap_report())
    monkeypatch.setattr(app_module, "insert_history", _noop_insert_history)

    response = client.post("/gap-analysis", json={"jd_text": "Data Analyst role..."}, headers=CLIENT_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["job_title"] == "Data Analyst"
    assert body["matched"][0]["requirement"] == "SQL"
    assert body["missing"][0]["requirement"] == "Tableau"
    assert body["overall_fit_summary"] == "Strong fit on core skills."


def test_gap_analysis_persists_a_history_record_on_success(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "run_parse_jd", lambda jd_text, settings: _jd())
    monkeypatch.setattr(app_module, "run_retrieve_evidence", lambda jd, settings: [_bundle()])
    monkeypatch.setattr(app_module, "run_gap_analysis", lambda jd, bundles, settings: _gap_report())
    monkeypatch.setattr(
        app_module, "insert_history", lambda db_path, client_id, kind, job_title, jd_text, result: calls.append(
            (client_id, kind, job_title, jd_text)
        )
    )

    client.post("/gap-analysis", json={"jd_text": "some JD"}, headers=CLIENT_HEADERS)
    assert calls == [("test-client", "gap_analysis", "Data Analyst", "some JD")]


def test_gap_analysis_requires_client_id_header():
    response = client.post("/gap-analysis", json={"jd_text": "x"})
    assert response.status_code == 400


def test_gap_analysis_rejects_empty_jd_text_with_422():
    response = client.post("/gap-analysis", json={"jd_text": ""}, headers=CLIENT_HEADERS)
    assert response.status_code == 422


def test_gap_analysis_maps_structured_output_error_to_502(monkeypatch):
    def _boom(jd_text, settings):
        raise StructuredOutputError("gave up after 3 attempts")

    monkeypatch.setattr(app_module, "run_parse_jd", _boom)
    response = client.post("/gap-analysis", json={"jd_text": "Data Analyst role..."}, headers=CLIENT_HEADERS)
    assert response.status_code == 502
    assert "gave up" in response.json()["detail"]


def test_gap_analysis_maps_missing_api_key_runtime_error_to_500(monkeypatch):
    def _boom(jd_text, settings):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr(app_module, "run_parse_jd", _boom)
    response = client.post("/gap-analysis", json={"jd_text": "Data Analyst role..."}, headers=CLIENT_HEADERS)
    assert response.status_code == 500


def test_draft_always_pauses_for_review_and_never_persists_history(monkeypatch):
    monkeypatch.setattr(
        app_module.agent_graph, "invoke", lambda payload, config: _pending_graph_result()
    )
    calls = []
    monkeypatch.setattr(
        app_module, "insert_history", lambda *a, **k: calls.append((a, k)),
    )

    response = client.post("/draft", json={"jd_text": "Data Analyst role..."}, headers=CLIENT_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_review"
    assert body["job_title"] == "Data Analyst"
    assert body["critic_passed"] is True
    assert body["revision_count"] == 0
    assert body["claims"][0]["evidence_chunk_ids"] == ["skills::chunk0"]
    assert body["gap_summary"] == "Strong fit on core skills."
    assert body["thread_id"]  # a real UUID was generated
    assert calls == []  # a pending draft is not a finished one -- nothing recorded yet


def test_draft_maps_structured_output_error_to_502(monkeypatch):
    def _boom(payload, config):
        raise StructuredOutputError("gave up after 3 attempts")

    monkeypatch.setattr(app_module.agent_graph, "invoke", _boom)
    response = client.post("/draft", json={"jd_text": "Data Analyst role..."}, headers=CLIENT_HEADERS)
    assert response.status_code == 502


def test_draft_decision_approve_finalizes_and_persists_history(monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module.agent_graph, "get_state", lambda config: _FakeStateSnapshot(_pending_graph_result())
    )
    monkeypatch.setattr(
        app_module.agent_graph, "invoke", lambda command, config: _approved_graph_result()
    )
    monkeypatch.setattr(
        app_module, "insert_history",
        lambda db_path, client_id, kind, job_title, jd_text, result: calls.append(
            (client_id, kind, job_title, jd_text)
        ),
    )

    response = client.post(
        "/draft/some-thread-id/decision", json={"action": "approve"}, headers=CLIENT_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["job_title"] == "Data Analyst"
    assert body["gap_summary"] is None  # only the pending step carries this
    assert calls == [("test-client", "draft", "Data Analyst", "Data Analyst role...")]


def test_draft_decision_revise_stays_pending_and_forwards_feedback(monkeypatch):
    received_commands = []

    def fake_invoke(command, config):
        received_commands.append(command)
        return _pending_graph_result(revision_count=1)

    monkeypatch.setattr(
        app_module.agent_graph, "get_state", lambda config: _FakeStateSnapshot(_pending_graph_result())
    )
    monkeypatch.setattr(app_module.agent_graph, "invoke", fake_invoke)
    monkeypatch.setattr(app_module, "insert_history", _noop_insert_history)

    response = client.post(
        "/draft/some-thread-id/decision",
        json={"action": "revise", "feedback": "Make it shorter."},
        headers=CLIENT_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_review"
    assert body["revision_count"] == 1
    assert received_commands[0].resume == {"action": "revise", "feedback": "Make it shorter."}


def test_draft_decision_unknown_thread_returns_404(monkeypatch):
    monkeypatch.setattr(app_module.agent_graph, "get_state", lambda config: _FakeStateSnapshot({}))
    response = client.post(
        "/draft/does-not-exist/decision", json={"action": "approve"}, headers=CLIENT_HEADERS,
    )
    assert response.status_code == 404


def test_draft_decision_maps_structured_output_error_to_502(monkeypatch):
    def _boom(command, config):
        raise StructuredOutputError("gave up after 3 attempts")

    monkeypatch.setattr(
        app_module.agent_graph, "get_state", lambda config: _FakeStateSnapshot(_pending_graph_result())
    )
    monkeypatch.setattr(app_module.agent_graph, "invoke", _boom)
    response = client.post(
        "/draft/some-thread-id/decision", json={"action": "approve"}, headers=CLIENT_HEADERS,
    )
    assert response.status_code == 502


def test_history_list_scopes_to_x_client_id_header(monkeypatch):
    captured = {}

    def fake_list_history(db_path, client_id):
        captured["client_id"] = client_id
        return [
            HistoryRecord(
                id="1", client_id=client_id, kind="draft", job_title="Data Analyst",
                jd_text="jd", result={}, created_at="2026-01-01T00:00:00+00:00",
            )
        ]

    monkeypatch.setattr(app_module, "list_history", fake_list_history)
    response = client.get("/history", headers=CLIENT_HEADERS)
    assert response.status_code == 200
    assert captured["client_id"] == "test-client"
    body = response.json()
    assert body == [{"id": "1", "kind": "draft", "job_title": "Data Analyst", "created_at": "2026-01-01T00:00:00+00:00"}]


def test_history_detail_returns_404_when_record_not_found(monkeypatch):
    monkeypatch.setattr(app_module, "get_history", lambda db_path, client_id, record_id: None)
    response = client.get("/history/does-not-exist", headers=CLIENT_HEADERS)
    assert response.status_code == 404


def test_history_detail_returns_full_record(monkeypatch):
    record = HistoryRecord(
        id="abc", client_id="test-client", kind="gap_analysis", job_title="Data Analyst",
        jd_text="the JD", result={"overall_fit_summary": "great"}, created_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(app_module, "get_history", lambda db_path, client_id, record_id: record)
    response = client.get("/history/abc", headers=CLIENT_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["jd_text"] == "the JD"
    assert body["result"] == {"overall_fit_summary": "great"}


def test_history_requires_client_id_header():
    response = client.get("/history")
    assert response.status_code == 400


# --- access-code gate (api/auth.py) ---
# get_settings is overridden via FastAPI's dependency_overrides (the standard testing
# pattern), not monkeypatch -- require_access_code takes it as Depends(get_settings),
# so this is the mechanism that's actually meant to be swapped in tests. Always
# cleaned up in `finally` so one test's override can't leak into the next.


def _with_access_code(code: str):
    protected = dataclasses.replace(real_get_settings(), access_code=code)
    app_module.app.dependency_overrides[real_get_settings] = lambda: protected


def _clear_access_code_override():
    app_module.app.dependency_overrides.pop(real_get_settings, None)


def test_auth_verify_ok_when_no_access_code_configured():
    # Default test environment has no ACCESS_CODE set (see config.py) -- gate is off.
    response = client.post("/auth/verify")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_auth_verify_rejects_missing_or_wrong_code_when_configured():
    _with_access_code("secret123")
    try:
        assert client.post("/auth/verify").status_code == 401
        assert client.post("/auth/verify", headers={"X-Access-Code": "wrong"}).status_code == 401
        r = client.post("/auth/verify", headers={"X-Access-Code": "secret123"})
        assert r.status_code == 200
    finally:
        _clear_access_code_override()


def test_gap_analysis_blocked_without_correct_access_code_when_configured(monkeypatch):
    monkeypatch.setattr(app_module, "run_parse_jd", lambda jd_text, settings: _jd())
    monkeypatch.setattr(app_module, "run_retrieve_evidence", lambda jd, settings: [_bundle()])
    monkeypatch.setattr(app_module, "run_gap_analysis", lambda jd, bundles, settings: _gap_report())
    monkeypatch.setattr(app_module, "insert_history", _noop_insert_history)

    _with_access_code("secret123")
    try:
        no_code = client.post("/gap-analysis", json={"jd_text": "x"}, headers=CLIENT_HEADERS)
        assert no_code.status_code == 401

        wrong_code = client.post(
            "/gap-analysis", json={"jd_text": "x"}, headers={**CLIENT_HEADERS, "X-Access-Code": "nope"}
        )
        assert wrong_code.status_code == 401

        right_code = client.post(
            "/gap-analysis", json={"jd_text": "x"}, headers={**CLIENT_HEADERS, "X-Access-Code": "secret123"}
        )
        assert right_code.status_code == 200
    finally:
        _clear_access_code_override()



def test_rejects_decision_from_a_different_client(monkeypatch):
    state = _pending_graph_result()
    state["client_id"] = "different-client"
    monkeypatch.setattr(
        app_module.agent_graph,
        "get_state",
        lambda config: _FakeStateSnapshot(state),
    )

    response = client.post(
        "/draft/some-thread-id/decision",
        json={"action": "approve"},
        headers=CLIENT_HEADERS,
    )

    assert response.status_code == 404


def test_rejects_oversized_job_description():
    response = client.post(
        "/gap-analysis",
        json={"jd_text": "x" * 30001},
        headers=CLIENT_HEADERS,
    )

    assert response.status_code == 422


def test_metrics_uses_route_template_for_dynamic_history_path(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_history",
        lambda db_path, client_id, record_id: None,
    )
    before = (
        client.get("/metrics")
        .json()["routes"]
        .get("/history/{record_id}", {})
        .get("count", 0)
    )

    response = client.get(
        "/history/metrics-template-test",
        headers=CLIENT_HEADERS,
    )
    assert response.status_code == 404

    routes = client.get("/metrics").json()["routes"]
    assert routes.get("/history/{record_id}", {}).get("count", 0) == before + 1
    assert "/history/metrics-template-test" not in routes
