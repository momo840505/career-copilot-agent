from __future__ import annotations

from typing import TypedDict

from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements


class AgentState(TypedDict, total=False):
    jd_text: str
    client_id: str
    jd: JDRequirements
    evidence_bundles: list[EvidenceBundle]
    gap_report: GapReport
    draft: CoverLetterDraft
    critic_verdict: CriticVerdict
    feedback_history: list[str]
    revision_count: int
    attempt_count: int
    human_decision: dict
