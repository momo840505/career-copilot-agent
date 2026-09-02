"""Phase 4b: wire the Phase 2-4a nodes into an actual LangGraph StateGraph.

Everything here reuses the plain-Python functions built in Phase 2-4a
(parse_jd, retrieve_evidence, gap_analysis, draft_writer, critic) completely
unchanged — this file only adds the state machine around them:
- the draft_writer <-> critic loop that draft_and_critique_demo.py drove by
  hand with a Python for-loop becomes a real conditional edge, bounded by
  the same MAX_REVISIONS safety net;
- a human_review node that actually pauses the graph via interrupt() instead
  of the "next: human_review" comment the architecture diagram has had since
  Phase 0.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from career_copilot.graph.critic import critic as run_critic
from career_copilot.graph.draft_writer import draft_writer as run_draft_writer
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.retrieve_evidence import EvidenceBundle, retrieve_evidence as run_retrieve_evidence
from career_copilot.graph.state import AgentState
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict, UngroundedClaim
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport
from career_copilot.schemas.jd import JDRequirements

MAX_REVISIONS = 2

# Every custom (non-builtin) type that ever ends up as a value in AgentState, so the
# checkpointer's serializer can reconstruct them by name on deserialize. Without this,
# InMemorySaver still works, but logs "Deserializing unregistered type ... this will be
# blocked in a future version" for each one — LangGraph's checkpoint msgpack format can
# reconstruct arbitrary Python objects, which is a real code-execution risk if a
# checkpoint store is ever attacker-controlled (see GHSA-g48c-2wqr-h844), so recent
# versions warn on any type that isn't explicitly allow-listed and will eventually
# refuse to load it at all. Explicitly listing our own trusted classes here (rather
# than leaving the default `allowed_msgpack_modules=True`, which allows anything) means
# this stays a warning-free no-op for us today and doesn't quietly break when a future
# langgraph version flips the default to strict.
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
    return {"draft": draft}


def _accumulate_feedback(history: list[str] | None, new_items: list[str]) -> list[str]:
    """Same dedup-and-append rule draft_and_critique_demo.py used by hand in Phase 4a
    — pulled out here so both _node_critic and _node_human_review can share it, and so
    it's independently testable without an API key (see tests/test_graph_routing.py)."""
    merged = list(history or [])
    for item in new_items:
        if item not in merged:
            merged.append(item)
    return merged


def _node_critic(state: AgentState) -> dict:
    verdict = run_critic(state["draft"], state["gap_report"], state["evidence_bundles"])
    updates: dict = {"critic_verdict": verdict}
    if not verdict.passed:
        new_feedback = verdict.issues + [
            f'Claim "{c.claim_text}" is not well-grounded: {c.reason}' for c in verdict.ungrounded_claims
        ]
        updates["feedback_history"] = _accumulate_feedback(state.get("feedback_history"), new_feedback)
        updates["revision_count"] = state.get("revision_count", 0) + 1
    return updates


def route_after_critic(state: AgentState) -> str:
    """Replaces draft_and_critique_demo.py's manual for-loop with a real conditional
    edge. Bounded exactly the same way: once revision_count hits MAX_REVISIONS, stop
    looping and send whatever we have to the human instead of retrying forever — a
    human reviewing an imperfect draft is a fine fallback, an infinite LLM loop is not.
    """
    verdict = state["critic_verdict"]
    if verdict.passed:
        return "human_review"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "human_review"
    return "draft_writer"


def _node_human_review(state: AgentState) -> dict:
    """The actual human-in-the-loop pause. `interrupt()` suspends the graph right here
    and hands `payload` back to whoever called .invoke()/.stream() — execution only
    resumes when someone calls `graph.invoke(Command(resume=decision), config=...)`
    with the SAME thread_id.

    Important LangGraph gotcha: on resume, this node function reruns from the top —
    everything written before the interrupt() call runs again too. That's harmless
    here (`payload` is just a read of existing state), but it's exactly why a node
    with a real side effect (an API call, a DB write, sending an email) belongs AFTER
    interrupt(), never before it — otherwise a resume silently repeats that side effect.
    """
    payload = {
        "draft": state["draft"].model_dump(),
        "critic_passed": state["critic_verdict"].passed,
        "critic_issues": state["critic_verdict"].issues,
        "gap_summary": state["gap_report"].overall_fit_summary,
    }
    decision = interrupt(payload)
    # Expected shape from the caller: {"action": "approve"} or
    # {"action": "revise", "feedback": "<free text>"}.
    updates: dict = {"human_decision": decision}
    if decision.get("action") == "revise" and decision.get("feedback"):
        note = f"Human reviewer feedback: {decision['feedback']}"
        updates["feedback_history"] = _accumulate_feedback(state.get("feedback_history"), [note])
    return updates


def route_after_human(state: AgentState) -> str:
    decision = state.get("human_decision") or {}
    if decision.get("action") == "approve":
        return END
    return "draft_writer"  # explicit "revise", or anything malformed -> try again


def build_graph():
    """Compile the full parse_jd -> retrieve_evidence -> gap_analysis -> draft_writer
    <-> critic -> human_review pipeline. Call this once per process; `graph.invoke()`
    takes a fresh `{"jd_text": ...}` and a `config={"configurable": {"thread_id": ...}}`
    per run — the thread_id is what lets a later Command(resume=...) find its way back
    to the same paused run.
    """
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
