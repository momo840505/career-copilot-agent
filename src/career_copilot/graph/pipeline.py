"""Unattended parse, retrieval, gap-analysis, draft, and critic pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from career_copilot.config import Settings, get_settings
from career_copilot.graph.critic import critic as run_critic
from career_copilot.graph.critic_feedback import accumulate_feedback, verdict_to_feedback_items
from career_copilot.graph.draft_writer import draft_writer as run_draft_writer
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.graph.retrieve_evidence import retrieve_evidence as run_retrieve_evidence
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements

MAX_REVISIONS = 2


@dataclass
class PipelineResult:
    jd: JDRequirements
    evidence_bundles: list[EvidenceBundle]
    gap_report: GapReport
    draft: CoverLetterDraft
    critic_verdict: CriticVerdict
    revision_count: int


def run_pipeline(
    jd_text: str,
    settings: Settings | None = None,
    max_revisions: int = MAX_REVISIONS,
) -> PipelineResult:
    settings = settings or get_settings()
    jd = run_parse_jd(jd_text, settings=settings)
    bundles = run_retrieve_evidence(jd, settings=settings)
    gap_report = run_gap_analysis(jd, bundles, settings=settings)

    feedback_history: list[str] = []
    draft: CoverLetterDraft | None = None
    verdict: CriticVerdict | None = None
    revision_count = 0

    for attempt in range(1, max_revisions + 2):
        draft = run_draft_writer(
            jd, gap_report, bundles, revision_feedback=feedback_history or None, settings=settings
        )
        verdict = run_critic(draft, gap_report, bundles, settings=settings)
        # Revision count excludes the initial draft.
        revision_count = attempt - 1
        if verdict.passed:
            break
        if attempt == max_revisions + 1:
            break
        feedback_history = accumulate_feedback(
            feedback_history, verdict_to_feedback_items(verdict)
        )

    assert draft is not None and verdict is not None
    return PipelineResult(
        jd=jd,
        evidence_bundles=bundles,
        gap_report=gap_report,
        draft=draft,
        critic_verdict=verdict,
        revision_count=revision_count,
    )
