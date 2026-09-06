"""Pure-logic tests for the MCP server's tool functions -- no API key needed.

Each @mcp.tool()-decorated function in career_copilot.mcp_server is a thin wrapper over
a pipeline function imported by name into that module (run_search, run_parse_jd,
run_retrieve_evidence, run_gap_analysis, run_pipeline). Monkeypatching those names —
same approach as test_api.py — lets us verify the wrapper's dict-shaping logic without
ever calling OpenAI. FastMCP/MCPServer wraps each function in a Tool object, but the
original function is still directly callable in-process, which is all these tests need.

Needs the `mcp[cli]` package installed (see requirements.txt) since importing this
module imports `mcp.server`.
"""
from __future__ import annotations

from career_copilot import mcp_server
from career_copilot.graph.pipeline import PipelineResult
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport
from career_copilot.schemas.jd import JDRequirements


def _jd() -> JDRequirements:
    return JDRequirements(
        job_title="Data Analyst",
        company="Acme",
        seniority="mid",
        must_have_skills=["SQL"],
        keywords=["SQL"],
        summary="Analyze data.",
    )


def _bundle() -> EvidenceBundle:
    chunk = RetrievedChunk(
        chunk_id="skills::chunk0", doc_id="doc", doc_title="Resume", text="Used SQL daily.", distance=0.1
    )
    return EvidenceBundle(requirement="SQL", chunks=[chunk])


def _gap_report() -> GapReport:
    return GapReport(
        matched=[GapItem(requirement="SQL", evidence_chunk_ids=["skills::chunk0"], note="Direct match.")],
        overall_fit_summary="Strong fit.",
    )


def _draft() -> CoverLetterDraft:
    return CoverLetterDraft(
        greeting="Dear Hiring Manager,",
        body="I have used SQL extensively.",
        closing="Sincerely,",
        claims=[Claim(text="I have used SQL extensively.", evidence_chunk_ids=["skills::chunk0"])],
    )


def _tool_fn(tool):
    """Support both a plain callable and an SDK-wrapped Tool object transparently —
    whichever shape the installed mcp[cli] version hands back from the @mcp.tool()
    decorator, the original function is reachable one way or the other."""
    return tool if callable(tool) else tool.fn


def test_search_evidence_shapes_retrieved_chunks_as_dicts(monkeypatch):
    monkeypatch.setattr(mcp_server, "run_search", lambda settings, query, k: [_bundle().chunks[0]])
    result = _tool_fn(mcp_server.search_evidence)("SQL", k=4)
    assert result == [
        {"chunk_id": "skills::chunk0", "doc_title": "Resume", "text": "Used SQL daily.", "distance": 0.1}
    ]


def test_analyze_job_description_shapes_gap_report_as_dict(monkeypatch):
    monkeypatch.setattr(mcp_server, "run_parse_jd", lambda jd_text, settings: _jd())
    monkeypatch.setattr(mcp_server, "run_retrieve_evidence", lambda jd, settings: [_bundle()])
    monkeypatch.setattr(mcp_server, "run_gap_analysis", lambda jd, bundles, settings: _gap_report())

    result = _tool_fn(mcp_server.analyze_job_description)("Data Analyst role...")
    assert result["job_title"] == "Data Analyst"
    assert result["matched"][0]["requirement"] == "SQL"
    assert result["overall_fit_summary"] == "Strong fit."


def test_draft_cover_letter_shapes_pipeline_result_as_dict(monkeypatch):
    pipeline_result = PipelineResult(
        jd=_jd(),
        evidence_bundles=[_bundle()],
        gap_report=_gap_report(),
        draft=_draft(),
        critic_verdict=CriticVerdict(passed=True),
        revision_count=1,
    )
    monkeypatch.setattr(mcp_server, "run_pipeline", lambda jd_text, settings: pipeline_result)

    result = _tool_fn(mcp_server.draft_cover_letter)("Data Analyst role...")
    assert result["job_title"] == "Data Analyst"
    assert result["draft"]["body"] == "I have used SQL extensively."
    assert result["critic_passed"] is True
    assert result["revision_count"] == 1
