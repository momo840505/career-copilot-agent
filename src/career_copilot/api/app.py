"""FastAPI entry point for the Career Copilot workflow."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from openai import APIError
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

# A process-level graph preserves in-memory review checkpoints between requests.
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



@app.middleware("http")
async def log_and_record_requests(request: Request, call_next):
    """Record request latency and status using the matched route template."""
    started_at = time.monotonic()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.monotonic() - started_at) * 1000
        status_code = response.status_code if response is not None else 500
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        metrics.record_request(route_path, status_code, duration_ms)
        logger.info(
            "%s %s -> %d (%.0fms)",
            request.method,
            route_path,
            status_code,
            duration_ms,
            extra={
                "method": request.method,
                "path": route_path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 1),
            },
        )


# HTTP request and response models are separate from internal structured-output schemas.


MAX_JD_CHARS = 30_000
MAX_FEEDBACK_CHARS = 4_000


class JDTextRequest(BaseModel):
    jd_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_JD_CHARS,
        description="Raw job description text.",
    )


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
        max_length=MAX_FEEDBACK_CHARS,
        description='Optional revision feedback when action="revise".',
    )


class DraftStepResponse(BaseModel):
    """Response shared by draft creation and review decisions."""

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
    """Public deployment health probe."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics_snapshot() -> dict:
    """Return process-local aggregate request metrics without user content."""
    return metrics.snapshot()


@app.post("/auth/verify")
def auth_verify(
    _rl: None = Depends(rate_limit(5)),
    _: None = Depends(require_access_code),
) -> dict:
    """Validate the shared demo access code under the auth rate limit."""
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
    except (StructuredOutputError, APIError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        # Runtime configuration errors are server-side failures.
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
    """Build a response for a draft paused at human review."""
    interrupt_payload = result["__interrupt__"][0].value
    draft = interrupt_payload["draft"]
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
    """Build a response for an approved draft."""
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
    """Start the LangGraph drafting workflow and pause at human review."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent_graph.invoke(
            {"jd_text": request.jd_text, "client_id": client_id}, config=config
        )
    except (StructuredOutputError, APIError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Preserve a valid response if a future graph version reaches END immediately.
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
    """Resume a paused draft with an approve or revise decision."""
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = agent_graph.get_state(config)
    if not snapshot.values or snapshot.values.get("client_id") != client_id:
        raise HTTPException(
            status_code=404,
            detail="Review session not found or has expired. Please start a new draft.",
        )

    decision: dict = {"action": request.action}
    if request.action == "revise":
        decision["feedback"] = request.feedback or "Please revise and try again."

    try:
        result = agent_graph.invoke(Command(resume=decision), config=config)
    except (StructuredOutputError, APIError) as e:
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
    """Return history metadata without the full saved payload."""
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


# Register the production frontend last so API routes keep precedence.
_frontend_dist = os.getenv("FRONTEND_DIST_DIR")
if _frontend_dist and Path(_frontend_dist).is_dir():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
