"""Phase 4b routing-logic tests — pure Python control flow, no API key needed.
(These do need langgraph installed, same as every other career_copilot.graph.*
module — that's a real dependency, not an optional extra.)
"""
from langgraph.graph import END

from career_copilot.graph.build_graph import (
    MAX_REVISIONS,
    _accumulate_feedback,
    _ALLOWED_CHECKPOINT_TYPES,
    _make_checkpointer,
    route_after_critic,
    route_after_human,
)
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements


def _verdict(passed: bool, **kwargs) -> CriticVerdict:
    return CriticVerdict(passed=passed, **kwargs)


def test_route_after_critic_passes_goes_to_human_review():
    state = {"critic_verdict": _verdict(True), "revision_count": 0}
    assert route_after_critic(state) == "human_review"


def test_route_after_critic_fails_under_max_goes_back_to_writer():
    state = {"critic_verdict": _verdict(False, issues=["too generic"]), "revision_count": 0}
    assert route_after_critic(state) == "draft_writer"


def test_route_after_critic_fails_at_max_goes_to_human_review():
    state = {"critic_verdict": _verdict(False, issues=["still bad"]), "revision_count": MAX_REVISIONS}
    assert route_after_critic(state) == "human_review"


def test_route_after_human_approve_ends():
    state = {"human_decision": {"action": "approve"}}
    assert route_after_human(state) == END


def test_route_after_human_revise_goes_back_to_writer():
    state = {"human_decision": {"action": "revise", "feedback": "too formal"}}
    assert route_after_human(state) == "draft_writer"


def test_route_after_human_missing_decision_defaults_to_revise():
    # Defensive: an empty/malformed decision should never be silently treated as
    # approval — better to loop back than to ship something nobody actually approved.
    assert route_after_human({}) == "draft_writer"


def test_accumulate_feedback_dedupes_and_preserves_order():
    history = ["a", "b"]
    result = _accumulate_feedback(history, ["b", "c"])
    assert result == ["a", "b", "c"]
    assert history == ["a", "b"]  # original list not mutated in place


def test_accumulate_feedback_from_none():
    assert _accumulate_feedback(None, ["x"]) == ["x"]


def test_checkpointer_allowlist_covers_every_agentstate_custom_type():
    # These are exactly the 6 classes that showed up as "Deserializing unregistered
    # type" warnings on the first live run — locking in that every custom type stored
    # in AgentState is explicitly allow-listed for the checkpointer's serializer.
    required = {JDRequirements, RetrievedChunk, EvidenceBundle, GapReport, CoverLetterDraft, CriticVerdict}
    assert required.issubset(set(_ALLOWED_CHECKPOINT_TYPES))


def test_make_checkpointer_builds_without_error():
    checkpointer = _make_checkpointer()
    assert checkpointer is not None
