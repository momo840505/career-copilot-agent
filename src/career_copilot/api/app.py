"""Phase 6/7: FastAPI service. Same pipeline the CLI demos, MCP server, and eval
harness all call — this is just another entry point over it, not a separate
implementation.

Run with: python scripts/run_api.py   (docs at http://127.0.0.1:8000/docs)

Error mapping: StructuredOutputError means the LLM never produced valid, rule-passing
output after every repair attempt in invoke_structured's bounded retry loop was
exhausted (see graph/structured.py). That's not a bad request from the client — the
request was fine, an *upstream dependency* (the LLM) failed to deliver — so it's mapped
to 502 Bad Gateway, not 400/422. A missing/invalid OPENAI_API_KEY is a server
misconfiguration, mapped to 500.

Phase 7 additions: every route except /health and /auth/verify requires the shared
access code (api/auth.py) once ACCESS_CODE is set, and /gap-analysis + /draft now
persist a record of each successful call to SQLite (api/db.py) for the frontend's
history view.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from career_copilot.api.auth import get_client_id, require_access_code
from career_copilot.api.db import get_history, init_db, insert_history, list_history
from career_copilot.config import get_settings
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.pipeline import run_pipeline
from career_copilot.graph.retrieve_evidence import retrieve_evidence as run_retrieve_evidence
from career_copilot.graph.structured import StructuredOutputError
from career_copilot.schemas.draft import Claim
from career_copilot.schemas.gap import GapItem


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(get_settings().history_db_path)
    yield


app = FastAPI(
    title="career-copilot-agent",
    description="JD -> gap analysis -> cover letter draft, with enforced citations and self-correction.",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS is fine here: the real access boundary is the shared access code
# (api/auth.py), not same-origin policy, and the frontend sends its credential as a
# plain custom header (X-Access-Code), never a cookie, so allow_credentials stays
# False and a wildcard origin can't be abused to steal a session the way it could with
# cookie-based auth. Needed for local dev, where the Vite dev server (a different
# origin/port) talks to this API directly; in the Phase 7c Docker image the frontend
# is served by this same app, so it's same-origin there and this middleware is a
# no-op in practice.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


class HistorySummary(BaseModel):
    id: str
    kind: str
    job_title: str
    created_at: str


class HistoryDetail(HistorySummary):
    jd_text: str
    result: dict


@app.get("/health")
def health() -> dict:
    """No LLM call, no access code required — deployment platforms probe this without
    any custom header, and it's useful to be able to check the service is up even
    without the code in hand."""
    settings = get_settings()
    return {"status": "ok", "api_key_configured": bool(settings.openai_api_key)}


@app.post("/auth/verify")
def auth_verify(_: None = Depends(require_access_code)) -> dict:
    """What the frontend's login screen calls to check a code before storing it —
    needs to be reachable WITHOUT already having a verified code, so it can't itself
    require one via any means other than the header being checked, i.e. this route's
    entire job is running require_access_code and reporting whether it raised."""
    return {"ok": True}


@app.post("/gap-analysis", response_model=GapAnalysisResponse)
def gap_analysis(
    request: JDTextRequest,
    _: None = Depends(require_access_code),
    client_id: str = Depends(get_client_id),
) -> GapAnalysisResponse:
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
    response = GapAnalysisResponse(
        job_title=jd.job_title,
        company=jd.company,
        seniority=jd.seniority,
        matched=report.matched,
        partial=report.partial,
        missing=report.missing,
        suggested_talking_points=report.suggested_talking_points,
        overall_fit_summary=report.overall_fit_summary,
    )
    insert_history(
        settings.history_db_path, client_id, "gap_analysis", jd.job_title, request.jd_text,
        response.model_dump(),
    )
    return response


@app.post("/draft", response_model=DraftResponse)
def draft(
    request: JDTextRequest,
    _: None = Depends(require_access_code),
    client_id: str = Depends(get_client_id),
) -> DraftResponse:
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
    response = DraftResponse(
        job_title=result.jd.job_title,
        greeting=result.draft.greeting,
        body=result.draft.body,
        closing=result.draft.closing,
        claims=result.draft.claims,
        critic_passed=result.critic_verdict.passed,
        critic_issues=result.critic_verdict.issues,
        revision_count=result.revision_count,
    )
    insert_history(
        settings.history_db_path, client_id, "draft", result.jd.job_title, request.jd_text,
        response.model_dump(),
    )
    return response


@app.get("/history", response_model=list[HistorySummary])
def history_list(
    _: None = Depends(require_access_code),
    client_id: str = Depends(get_client_id),
) -> list[HistorySummary]:
    """Summaries only (no jd_text/result) — the list view doesn't need the full
    payload, and keeping it light matters once someone has dozens of past runs."""
    records = list_history(get_settings().history_db_path, client_id)
    return [
        HistorySummary(id=r.id, kind=r.kind, job_title=r.job_title, created_at=r.created_at)
        for r in records
    ]


@app.get("/history/{record_id}", response_model=HistoryDetail)
def history_detail(
    record_id: str,
    _: None = Depends(require_access_code),
    client_id: str = Depends(get_client_id),
) -> HistoryDetail:
    record = get_history(get_settings().history_db_path, client_id, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found.")
    return HistoryDetail(
        id=record.id,
        kind=record.kind,
        job_title=record.job_title,
        created_at=record.created_at,
        jd_text=record.jd_text,
        result=record.result,
    )


# --- Phase 7c: serve the built frontend (if present) ---
# Must be registered LAST: Starlette matches routes in registration order, and a Mount
# at "/" matches every path, so every @app.get/@app.post route above needs to already
# be in app.router.routes before this runs, or the mount would shadow them.
#
# FRONTEND_DIST_DIR is unset in local dev (Vite's own dev server serves the frontend
# there instead — see frontend/vite.config.js's proxy) and set to /app/frontend_dist
# by the Docker image (see ../../../Dockerfile), so this mount is a no-op except in
# the built container. html=True serves frontend_dist/index.html for "/" — the SPA has
# no client-side routes of its own (App.jsx switches tabs via React state, not a
# router), so nothing else needs a fallback.
_frontend_dist = os.getenv("FRONTEND_DIST_DIR")
if _frontend_dist and Path(_frontend_dist).is_dir():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
