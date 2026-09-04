"""Phase 4b routing-logic tests — pure Python control flow, no API key needed.
(These do need langgraph installed, same as every other career_copilot.graph.*
module — that's a real dependency, not an optional extra.)
"""
from langgraph.graph import END

import career_copilot.graph.build_graph as build_graph_module
from career_copilot.graph.build_graph import (
    MAX_REVISIONS,
    _accumulate_feedback,
    _ALLOWED_CHECKPOINT_TYPES,
    _make_checkpointer,
    _node_critic,
    _node_draft_writer,
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


def test_draft_critic_loop_allows_the_full_max_revisions_budget(monkeypatch):
    """Regression: _node_critic used to increment revision_count only on failure,
    which stopped route_after_critic after just MAX_REVISIONS - 1 real revisions
    instead of the full MAX_REVISIONS budget (drift from graph/pipeline.py's own,
    already-fixed, already-tested loop bound for the exact same MAX_REVISIONS).
    Drives _node_draft_writer -> _node_critic -> route_after_critic by hand across
    every attempt the budget allows, all of them failing, with run_draft_writer/
    run_critic monkeypatched to fakes -- no API key needed, same approach as
    tests/test_pipeline_revision_count.py.
    """
    monkeypatch.setattr(build_graph_module, "run_draft_writer", lambda *a, **k: "draft")
    monkeypatch.setattr(
        build_graph_module, "run_critic", lambda *a, **k: _verdict(False, issues=["still bad"])
    )

    state: dict = {"jd": None, "gap_report": None, "evidence_bundles": None}
    routes = []
    for _ in range(MAX_REVISIONS + 1):  # 1 initial attempt + MAX_REVISIONS repairs
        state.update(_node_draft_writer(state))
        state.update(_node_critic(state))
        route = route_after_critic(state)
        routes.append(route)
        if route == "human_review":
            break

    # 3 total draft attempts for MAX_REVISIONS=2 (1 initial + 2 repairs) -- matching
    # pipeline.py's exhausted-budget case exactly, not one repair short of it.
    assert state["attempt_count"] == MAX_REVISIONS + 1
    assert state["revision_count"] == MAX_REVISIONS
    assert routes == ["draft_writer"] * MAX_REVISIONS + ["human_review"]


def test_draft_critic_loop_converges_on_the_last_attempt_the_budget_allows(monkeypatch):
    monkeypatch.setattr(build_graph_module, "run_draft_writer", lambda *a, **k: "draft")
    verdicts = [_verdict(False, issues=["still bad"])] * MAX_REVISIONS + [_verdict(True)]
    calls = {"n": 0}

    def fake_critic(*a, **k):
        v = verdicts[calls["n"]]
        calls["n"] += 1
        return v

    monkeypatch.setattr(build_graph_module, "run_critic", fake_critic)

    state: dict = {"jd": None, "gap_report": None, "evidence_bundles": None}
    route = None
    for _ in range(MAX_REVISIONS + 1):
        state.update(_node_draft_writer(state))
        state.update(_node_critic(state))
        route = route_after_critic(state)
        if route == "human_review":
            break

    assert route == "human_review"  # passed -> straight to human_review, not another retry
    assert state["critic_verdict"].passed is True
    assert state["revision_count"] == MAX_REVISIONS  # it took the full budget to converge


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
