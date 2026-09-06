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

Phase 7e addition: structured logging (career_copilot/observability.py) is configured
at import time, below -- before `app = FastAPI(...)` runs, so even startup-time log
lines (e.g. from init_db in lifespan) go through it. A request-logging middleware and
GET /metrics expose the same module's in-process counters.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from career_copilot.api.auth import get_client_id, require_access_code
from career_copilot.api.db import get_history, init_db, insert_history, list_history
from career_copilot.api.rate_limit import rate_limit
from career_copilot.config import get_settings
from career_copilot.graph.build_graph import build_graph
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.retrieve_evidence import retrieve_evidence as run_retrieve_evidence
from career_copilot.graph.structured import StructuredOutputError
from career_copilot.observability import configure_logging, metrics
from career_copilot.schemas.draft import Claim
from career_copilot.schemas.gap import GapItem

configure_logging()
logger = logging.getLogger(__name__)

# Compiled once per process, not once per request -- build_graph()'s own docstring
# is explicit about this ("call this once per process"). Its checkpointer (an
# InMemorySaver, see build_graph.py) is what lets POST /draft/{thread_id}/decision
# find its way back to a specific paused run: the graph module stays the same object
# across requests, so the checkpoints it wrote during /draft are still there when
# /draft/{thread_id}/decision resumes them. This also means a paused (pending-review)
# draft does not survive a process restart/redeploy -- acceptable for a single-
# instance portfolio demo, not for anything scaled beyond one worker process.
agent_graph = build_graph()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(get_settings().history_db_path)
    logger.info("startup complete", extra={"history_db": str(get_settings().history_db_path)})
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


@app.middleware("http")
async def log_and_record_requests(request: Request, call_next):
    """Every request through one place: one structured log line + one metrics update,
    regardless of which route handled it (or whether it 500'd). `request.url.path` is
    used as the metrics/log key rather than a route template (e.g. "/history/{id}")
    because FastAPI only resolves the matched route AFTER this middleware runs; that's
    fine at this traffic scale (history IDs are UUIDs, so they won't collapse into a
    misleadingly "popular" single bucket the way a numeric ID might, and a portfolio
    demo doesn't have enough unique history records for that to matter anyway).
    """
    started_at = time.monotonic()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.monotonic() - started_at) * 1000
        status_code = response.status_code if response is not None else 500
        metrics.record_request(request.url.path, status_code, duration_ms)
        logger.info(
            "%s %s -> %d (%.0fms)",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 1),
                "client_id": request.headers.get("x-client-id"),
            },
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


class DraftDecisionRequest(BaseModel):
    action: Literal["approve", "revise"]
    feedback: str | None = Field(
        default=None,
        description='Required in spirit (not enforced) for action="revise" -- see '
        "_node_human_review's fallback text if omitted.",
    )


class DraftStepResponse(BaseModel):
    """What both POST /draft and POST /draft/{thread_id}/decision return. The graph's
    human_review node (build_graph.py) always pauses after the draft/critic loop ends
    -- whether the critic passed or the revision budget ran out -- so a fresh POST
    /draft never finishes a letter by itself; status is always "pending_review" there.
    It only becomes "approved" from the decision route, once a human sends
    {"action": "approve"} and the graph reaches END with no further interrupt.
    """

    thread_id: str
    status: Literal["pending_review", "approved"]
    job_title: str
    greeting: str
    body: str
    closing: str
    claims: list[Claim]
    critic_passed: bool
    critic_issues: list[str]
    gap_summary: str | None = None
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


@app.get("/metrics")
def metrics_snapshot() -> dict:
    """No access code required, same reasoning as /health: this is an ops endpoint,
    not a data endpoint. It exposes aggregate counts and latencies only -- no JD text,
    no cover letters, no access codes, nothing tied to an individual client_id -- so
    unlike /gap-analysis, /draft, and /history, gating it behind the shared code would
    only make it harder to check the service's health, not protect anything sensitive.
    Resets to zero on every process restart (in-memory only -- see observability.py).
    """
    return metrics.snapshot()


@app.post("/auth/verify")
def auth_verify(
    _rl: None = Depends(rate_limit(5)),
    _: None = Depends(require_access_code),
) -> dict:
    """What the frontend's login screen calls to check a code before storing it —
    needs to be reachable WITHOUT already having a verified code, so it can't itself
    require one via any means other than the header being checked, i.e. this route's
    entire job is running require_access_code and reporting whether it raised.

    `rate_limit(5)` is listed BEFORE `require_access_code` on purpose: FastAPI
    resolves Depends() in declared order, so a wrong access code no longer gets a
    free pass on the limit by raising its 401 first — see api/rate_limit.py's
    docstring for how that hole was found and confirmed fixed.
    """
    return {"ok": True}


@app.post("/gap-analysis", response_model=GapAnalysisResponse)
def gap_analysis(
    request: JDTextRequest,
    _rl: None = Depends(rate_limit(10)),
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


def _pending_review_response(thread_id: str, result: dict) -> DraftStepResponse:
    """Build a DraftStepResponse from a graph result that just hit human_review's
    interrupt() -- `result["__interrupt__"][0].value` is exactly the `payload` dict
    _node_human_review (build_graph.py) passed to interrupt()."""
    interrupt_payload = result["__interrupt__"][0].value
    draft = interrupt_payload["draft"]  # already a plain dict (draft.model_dump())
    return DraftStepResponse(
        thread_id=thread_id,
        status="pending_review",
        job_title=result["jd"].job_title,
        greeting=draft["greeting"],
        body=draft["body"],
        closing=draft["closing"],
        claims=draft["claims"],
        critic_passed=interrupt_payload["critic_passed"],
        critic_issues=interrupt_payload["critic_issues"],
        gap_summary=interrupt_payload["gap_summary"],
        revision_count=result.get("revision_count", 0),
    )


def _approved_response(thread_id: str, result: dict) -> DraftStepResponse:
    """Build a DraftStepResponse from a graph result that reached END (the human
    approved and no interrupt is pending)."""
    return DraftStepResponse(
        thread_id=thread_id,
        status="approved",
        job_title=result["jd"].job_title,
        greeting=result["draft"].greeting,
        body=result["draft"].body,
        closing=result["draft"].closing,
        claims=result["draft"].claims,
        critic_passed=result["critic_verdict"].passed,
        critic_issues=result["critic_verdict"].issues,
        gap_summary=None,
        revision_count=result.get("revision_count", 0),
    )


@app.post("/draft", response_model=DraftStepResponse)
def draft(
    request: JDTextRequest,
    _rl: None = Depends(rate_limit(10)),
    _: None = Depends(require_access_code),
    client_id: str = Depends(get_client_id),
) -> DraftStepResponse:
    """Starts the full pipeline (parse -> retrieve -> gap analysis -> draft/critic
    loop) on the compiled LangGraph StateGraph (build_graph.py). The graph's
    human_review node always pauses here via interrupt() -- this call never returns
    a finished letter by itself. The response's `thread_id` is what the client sends
    back to POST /draft/{thread_id}/decision to approve or request a revision.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent_graph.invoke({"jd_text": request.jd_text}, config=config)
    except StructuredOutputError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # route_after_critic always sends the graph to human_review (pass or exhausted
    # revision budget), so this should always be true -- but if a future change to
    # the graph ever let it reach END on the very first call, degrade gracefully to
    # an "approved" response (and record history) rather than crashing on a KeyError.
    if "__interrupt__" not in result:
        response = _approved_response(thread_id, result)
        insert_history(
            get_settings().history_db_path, client_id, "draft", response.job_title,
            request.jd_text, response.model_dump(),
        )
        return response

    return _pending_review_response(thread_id, result)


@app.post("/draft/{thread_id}/decision", response_model=DraftStepResponse)
def draft_decision(
    thread_id: str,
    request: DraftDecisionRequest,
    _rl: None = Depends(rate_limit(10)),
    _: None = Depends(require_access_code),
    client_id: str = Depends(get_client_id),
) -> DraftStepResponse:
    """Resumes a paused draft (see POST /draft) with the human's decision.

    action="approve" ends the graph at END and returns the final letter -- that's
    the only path that persists a history record, mirroring the old (pre-review)
    /draft behavior of only ever recording a *finished* draft.

    action="revise" sends the graph back to draft_writer with the given feedback
    folded into feedback_history, and the response pauses at human_review again
    (another "pending_review"), same as the CLI's scripts/graph_demo.py.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if not agent_graph.get_state(config).values:
        raise HTTPException(
            status_code=404,
            detail="Review session not found or has expired. Please start a new draft.",
        )

    decision: dict = {"action": request.action}
    if request.action == "revise":
        decision["feedback"] = request.feedback or "Please revise and try again."

    try:
        result = agent_graph.invoke(Command(resume=decision), config=config)
    except StructuredOutputError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if "__interrupt__" in result:
        return _pending_review_response(thread_id, result)

    response = _approved_response(thread_id, result)
    insert_history(
        get_settings().history_db_path, client_id, "draft", response.job_title,
        result.get("jd_text", ""), response.model_dump(),
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
