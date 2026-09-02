"""Phase 6: FastAPI service. Same pipeline the CLI demos, MCP server, and eval harness
all call — this is just another entry point over it, not a separate implementation.

Run with: python scripts/run_api.py   (docs at http://127.0.0.1:8000/docs)

Error mapping: StructuredOutputError means the LLM never produced valid, rule-passing
output after every repair attempt in invoke_structured's bounded retry loop was
exhausted (see graph/structured.py). That's not a bad request from the client — the
request was fine, an *upstream dependency* (the LLM) failed to deliver — so it's mapped
to 502 Bad Gateway, not 400/422. A missing/invalid OPENAI_API_KEY is a server
misconfiguration, mapped to 500.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from career_copilot.config import get_settings
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.pipeline import run_pipeline
from career_copilot.graph.retrieve_evidence import retrieve_evidence as run_retrieve_evidence
from career_copilot.graph.structured import StructuredOutputError
from career_copilot.schemas.draft import Claim
from career_copilot.schemas.gap import GapItem

app = FastAPI(
    title="career-copilot-agent",
    description="JD -> gap analysis -> cover letter draft, with enforced citations and self-correction.",
    version="0.1.0",
)


# --- request/response models ---
# Deliberately separate from the internal schemas/*.py Pydantic models (which are the
# LLM's structured-output contract, tuned for prompting via Field descriptions) — these
# are the HTTP contract, and the two are allowed to drift independently.


class JDTextRequest(BaseModel):
    jd_text: str = Field(..., min_length=1, description="Raw job description text.")


class GapAnalysisResponse(BaseModel):
    job_title: str
    company: str | None
    seniority: str
    matched: list[GapItem]
    partial: list[GapItem]
    missing: list[GapItem]
    suggested_talking_points: list[str]
    overall_fit_summary: str


class DraftResponse(BaseModel):
    job_title: str
    greeting: str
    body: str
    closing: str
    claims: list[Claim]
    critic_passed: bool
    critic_issues: list[str]
    revision_count: int


@app.get("/health")
def health() -> dict:
    """No LLM call — just confirms the service is up and an API key is configured."""
    settings = get_settings()
    return {"status": "ok", "api_key_configured": bool(settings.openai_api_key)}


@app.post("/gap-analysis", response_model=GapAnalysisResponse)
def gap_analysis(request: JDTextRequest) -> GapAnalysisResponse:
    settings = get_settings()
    try:
        jd = run_parse_jd(request.jd_text, settings=settings)
        bundles = run_retrieve_evidence(jd, settings=settings)
        report = run_gap_analysis(jd, bundles, settings=settings)
    except StructuredOutputError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        # e.g. OPENAI_API_KEY not set — a server misconfiguration, not the client's fault.
        raise HTTPException(status_code=500, detail=str(e)) from e
    return GapAnalysisResponse(
        job_title=jd.job_title,
        company=jd.company,
        seniority=jd.seniority,
        matched=report.matched,
        partial=report.partial,
        missing=report.missing,
        suggested_talking_points=report.suggested_talking_points,
        overall_fit_summary=report.overall_fit_summary,
    )


@app.post("/draft", response_model=DraftResponse)
def draft(request: JDTextRequest) -> DraftResponse:
    """Runs the full pipeline (parse -> retrieve -> gap analysis -> draft/critic loop),
    same as scripts/run_evals.py — no human-in-the-loop approval here. The client is
    expected to review `critic_passed` / `critic_issues` itself before using the draft
    for anything, exactly as the README's "Known limitations" section describes.
    """
    settings = get_settings()
    try:
        result = run_pipeline(request.jd_text, settings=settings)
    except StructuredOutputError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return DraftResponse(
        job_title=result.jd.job_title,
        greeting=result.draft.greeting,
        body=result.draft.body,
        closing=result.draft.closing,
        claims=result.draft.claims,
        critic_passed=result.critic_verdict.passed,
        critic_issues=result.critic_verdict.issues,
        revision_count=result.revision_count,
    )
