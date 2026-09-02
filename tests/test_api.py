"""Pure-logic tests for the FastAPI service (Phase 6) — no API key needed.

Every route handler in career_copilot.api.app calls the pipeline functions by name
imported into that module's namespace (run_parse_jd, run_retrieve_evidence,
run_gap_analysis, run_pipeline), so monkeypatching those names there — the same
pattern as swapping a real LLM for a fake one in test_invoke_structured_retry.py —
exercises the real request/response wiring (validation, status codes, field mapping)
without ever calling OpenAI. Uses FastAPI's TestClient (httpx under the hood, already
pulled in transitively via langchain-openai -> openai's own httpx dependency).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from career_copilot.api import app as app_module
from career_copilot.graph.pipeline import PipelineResult
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.graph.structured import StructuredOutputError
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport
from career_copilot.schemas.jd import JDRequirements

client = TestClient(app_module.app)


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


def test_health_reports_ok_and_key_configured_flag(monkeypatch):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "api_key_configured" in body


def test_gap_analysis_maps_pipeline_result_to_response_fields(monkeypatch):
    monkeypatch.setattr(app_module, "run_parse_jd", lambda jd_text, settings: _jd())
    monkeypatch.setattr(app_module, "run_retrieve_evidence", lambda jd, settings: [_bundle()])
    monkeypatch.setattr(app_module, "run_gap_analysis", lambda jd, bundles, settings: _gap_report())

    response = client.post("/gap-analysis", json={"jd_text": "Data Analyst role..."})
    assert response.status_code == 200
    body = response.json()
    assert body["job_title"] == "Data Analyst"
    assert body["matched"][0]["requirement"] == "SQL"
    assert body["missing"][0]["requirement"] == "Tableau"
    assert body["overall_fit_summary"] == "Strong fit on core skills."


def test_gap_analysis_rejects_empty_jd_text_with_422():
    response = client.post("/gap-analysis", json={"jd_text": ""})
    assert response.status_code == 422


def test_gap_analysis_maps_structured_output_error_to_502(monkeypatch):
    def _boom(jd_text, settings):
        raise StructuredOutputError("gave up after 3 attempts")

    monkeypatch.setattr(app_module, "run_parse_jd", _boom)
    response = client.post("/gap-analysis", json={"jd_text": "Data Analyst role..."})
    assert response.status_code == 502
    assert "gave up" in response.json()["detail"]


def test_gap_analysis_maps_missing_api_key_runtime_error_to_500(monkeypatch):
    def _boom(jd_text, settings):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr(app_module, "run_parse_jd", _boom)
    response = client.post("/gap-analysis", json={"jd_text": "Data Analyst role..."})
    assert response.status_code == 500


def test_draft_maps_pipeline_result_to_response_fields(monkeypatch):
    result = PipelineResult(
        jd=_jd(),
        evidence_bundles=[_bundle()],
        gap_report=_gap_report(),
        draft=_draft(),
        critic_verdict=CriticVerdict(passed=True),
        revision_count=0,
    )
    monkeypatch.setattr(app_module, "run_pipeline", lambda jd_text, settings: result)

    response = client.post("/draft", json={"jd_text": "Data Analyst role..."})
    assert response.status_code == 200
    body = response.json()
    assert body["job_title"] == "Data Analyst"
    assert body["critic_passed"] is True
    assert body["revision_count"] == 0
    assert body["claims"][0]["evidence_chunk_ids"] == ["skills::chunk0"]


def test_draft_maps_structured_output_error_to_502(monkeypatch):
    def _boom(jd_text, settings):
        raise StructuredOutputError("gave up after 3 attempts")

    monkeypatch.setattr(app_module, "run_pipeline", _boom)
    response = client.post("/draft", json={"jd_text": "Data Analyst role..."})
    assert response.status_code == 502
