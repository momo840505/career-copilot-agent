"""Phase 4b: shared state schema for the LangGraph StateGraph.

InMemorySaver keeps checkpoints in this process's memory, so arbitrary Python
objects (Pydantic models, dataclasses) are fine as field values here. If this
ever moves to a persistent checkpointer (Sqlite/Postgres, so a run survives a
process restart — a Phase 6/7 concern) every field would need to become
JSON-serializable first; not needed yet.

`total=False` because the graph fills this in progressively: only `jd_text`
is present at the start, and each node only ever returns the few keys it's
responsible for (LangGraph merges a node's returned dict into the running
state — it isn't replacing the whole state each time).
"""
from __future__ import annotations

from typing import TypedDict

from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements


class AgentState(TypedDict, total=False):
    jd_text: str
    jd: JDRequirements
    evidence_bundles: list[EvidenceBundle]
    gap_report: GapReport
    draft: CoverLetterDraft
    critic_verdict: CriticVerdict
    feedback_history: list[str]
    revision_count: int
    human_decision: dict
