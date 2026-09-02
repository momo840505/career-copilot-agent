"""Phase 6: MCP server exposing career-copilot's pipeline as tools any MCP-aware client
(Claude Desktop, an IDE, another agent) can call directly, no HTTP client required.

Built against the mcp[cli] v2.x API. This matters: the SDK made a breaking change
between v1.x (`from mcp.server.fastmcp import FastMCP`) and v2.x (`from mcp.server import
MCPServer`, the class itself renamed) — see the comment on the `mcp[cli]` pin in
requirements.txt for how this was caught. Written and verified against v2's README
directly, not carried over from an older tutorial.

Run it with either:
    mcp dev src/career_copilot/mcp_server.py        # interactive inspector
    mcp run src/career_copilot/mcp_server.py --transport streamable-http

Each tool below is a thin wrapper over the same node functions the CLI demos, the eval
harness, and the FastAPI service all call — no logic is duplicated here, only exposed.
"""
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
    """Search the candidate's resume/portfolio for evidence relevant to a query
    (a skill, a requirement, a topic). Returns the top-k matching chunks, each with
    its chunk_id, source document title, the chunk text, and a similarity distance
    (lower = more relevant). Use this to check what real evidence exists BEFORE
    claiming the candidate has a given skill.
    """
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
    """Parse a raw job-description text into structured requirements, retrieve
    matching evidence from the candidate's portfolio for every requirement, and
    produce an honest gap analysis: which requirements are clearly matched, which
    are only partially/adjacently supported, and which have no supporting evidence
    at all. This is the same parse_jd -> retrieve_evidence -> gap_analysis pipeline
    used everywhere else in this project — nothing here is a separate, looser
    re-implementation.
    """
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
    """Run the FULL pipeline end to end — parse the JD, retrieve evidence, analyze
    gaps, then draft a cover letter through the draft/critic self-correction loop
    (bounded retries, same as everywhere else in this project) — and return the
    final draft plus whether the critic ultimately passed it and how many revisions
    it took. This has NO human-in-the-loop approval step (unlike graph_demo.py's
    interactive run) — it's meant for a caller that will review the draft itself
    before it's ever sent anywhere, exactly like scripts/run_evals.py.
    """
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
    # `python src/career_copilot/mcp_server.py` directly is NOT how the v2 SDK expects
    # this to run — there's no documented .run()/.serve() for that. Use the `mcp`
    # CLI instead (`mcp dev ...` / `mcp run ... --transport streamable-http`), per the
    # module docstring above.
    raise SystemExit(
        "Run this with the `mcp` CLI, not `python`: "
        "`mcp dev src/career_copilot/mcp_server.py` or "
        "`mcp run src/career_copilot/mcp_server.py --transport streamable-http`."
    )
