"""MCP tools for portfolio retrieval, job-gap analysis, and cover-letter drafting."""
from __future__ import annotations

from mcp.server import MCPServer

from career_copilot.config import get_settings
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.pipeline import run_pipeline
from career_copilot.graph.retrieve_evidence import retrieve_evidence as run_retrieve_evidence
from career_copilot.rag.retriever import search as run_search

mcp = MCPServer("career-copilot")


@mcp.tool()
def search_evidence(query: str, k: int = 4) -> list[dict]:
    """Return portfolio evidence relevant to a query."""
    settings = get_settings()
    chunks = run_search(settings, query, k=k)
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_title": c.doc_title,
            "text": c.text,
            "distance": c.distance,
        }
        for c in chunks
    ]


@mcp.tool()
def analyze_job_description(jd_text: str) -> dict:
    """Parse a job description and return an evidence-grounded gap analysis."""
    settings = get_settings()
    jd = run_parse_jd(jd_text, settings=settings)
    bundles = run_retrieve_evidence(jd, settings=settings)
    report = run_gap_analysis(jd, bundles, settings=settings)
    return {
        "job_title": jd.job_title,
        "company": jd.company,
        "seniority": jd.seniority,
        "matched": [item.model_dump() for item in report.matched],
        "partial": [item.model_dump() for item in report.partial],
        "missing": [item.model_dump() for item in report.missing],
        "suggested_talking_points": report.suggested_talking_points,
        "overall_fit_summary": report.overall_fit_summary,
    }


@mcp.tool()
def draft_cover_letter(jd_text: str) -> dict:
    """Run the unattended drafting pipeline and return the reviewed draft."""
    settings = get_settings()
    result = run_pipeline(jd_text, settings=settings)
    return {
        "job_title": result.jd.job_title,
        "draft": {
            "greeting": result.draft.greeting,
            "body": result.draft.body,
            "closing": result.draft.closing,
            "claims": [c.model_dump() for c in result.draft.claims],
        },
        "critic_passed": result.critic_verdict.passed,
        "critic_issues": result.critic_verdict.issues,
        "revision_count": result.revision_count,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Run with the MCP CLI: "
        "`mcp dev src/career_copilot/mcp_server.py` or "
        "`mcp run src/career_copilot/mcp_server.py --transport streamable-http`."
    )
