"""LangGraph workflow for evidence retrieval, drafting, critique, and human review."""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from career_copilot.graph.critic import critic as run_critic
from career_copilot.graph.critic_feedback import accumulate_feedback, verdict_to_feedback_items
from career_copilot.graph.draft_writer import draft_writer as run_draft_writer
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.graph.retrieve_evidence import retrieve_evidence as run_retrieve_evidence
from career_copilot.graph.state import AgentState
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict, UngroundedClaim
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport
from career_copilot.schemas.jd import JDRequirements

MAX_REVISIONS = 2

# Restrict checkpoint deserialization to the state types used by this graph.
_ALLOWED_CHECKPOINT_TYPES = [
    JDRequirements,
    RetrievedChunk,
    EvidenceBundle,
    GapReport,
    GapItem,
    CoverLetterDraft,
    Claim,
    CriticVerdict,
    UngroundedClaim,
]


def _make_checkpointer() -> InMemorySaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_CHECKPOINT_TYPES)
    return InMemorySaver(serde=serde)


def _node_parse_jd(state: AgentState) -> dict:
    return {"jd": run_parse_jd(state["jd_text"])}


def _node_retrieve_evidence(state: AgentState) -> dict:
    return {"evidence_bundles": run_retrieve_evidence(state["jd"])}


def _node_gap_analysis(state: AgentState) -> dict:
    return {"gap_report": run_gap_analysis(state["jd"], state["evidence_bundles"])}


def _node_draft_writer(state: AgentState) -> dict:
    draft = run_draft_writer(
        state["jd"],
        state["gap_report"],
        state["evidence_bundles"],
        revision_feedback=state.get("feedback_history") or None,
    )
    # attempt_count includes the initial draft; revision_count excludes it.
    attempt_count = state.get("attempt_count", 0) + 1
    return {"draft": draft, "attempt_count": attempt_count}


def _node_critic(state: AgentState) -> dict:
    verdict = run_critic(state["draft"], state["gap_report"], state["evidence_bundles"])
    updates: dict = {
        "critic_verdict": verdict,
        "revision_count": state.get("attempt_count", 1) - 1,
    }
    if not verdict.passed:
        updates["feedback_history"] = accumulate_feedback(
            state.get("feedback_history"), verdict_to_feedback_items(verdict)
        )
    return updates


def route_after_critic(state: AgentState) -> str:
    """Route an accepted or exhausted draft to human review."""
    verdict = state["critic_verdict"]
    if verdict.passed:
        return "human_review"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "human_review"
    return "draft_writer"


def _node_human_review(state: AgentState) -> dict:
    """Pause the graph until the matching thread is approved or revised."""
    payload = {
        "draft": state["draft"].model_dump(),
        "critic_passed": state["critic_verdict"].passed,
        "critic_issues": state["critic_verdict"].issues,
        "gap_summary": state["gap_report"].overall_fit_summary,
    }
    decision = interrupt(payload)
    # The caller supplies an approve or revise decision.
    updates: dict = {"human_decision": decision}
    if decision.get("action") == "revise" and decision.get("feedback"):
        note = f"Human reviewer feedback: {decision['feedback']}"
        updates["feedback_history"] = accumulate_feedback(state.get("feedback_history"), [note])
    return updates


def route_after_human(state: AgentState) -> str:
    decision = state.get("human_decision") or {}
    if decision.get("action") == "approve":
        return END
    return "draft_writer"


def build_graph():
    """Compile the web drafting workflow with an in-memory review checkpoint."""
    builder = StateGraph(AgentState)
    builder.add_node("parse_jd", _node_parse_jd)
    builder.add_node("retrieve_evidence", _node_retrieve_evidence)
    builder.add_node("gap_analysis", _node_gap_analysis)
    builder.add_node("draft_writer", _node_draft_writer)
    builder.add_node("critic", _node_critic)
    builder.add_node("human_review", _node_human_review)

    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "gap_analysis")
    builder.add_edge("gap_analysis", "draft_writer")
    builder.add_edge("draft_writer", "critic")
    builder.add_conditional_edges(
        "critic", route_after_critic, {"draft_writer": "draft_writer", "human_review": "human_review"}
    )
    builder.add_conditional_edges("human_review", route_after_human, {"draft_writer": "draft_writer", END: END})

    return builder.compile(checkpointer=_make_checkpointer())
