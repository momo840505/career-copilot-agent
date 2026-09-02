"""Phase 5: run the full parse_jd -> retrieve_evidence -> gap_analysis -> draft_writer
<-> critic pipeline to completion WITHOUT a human in the loop — for evals and any other
automated/batch use where nobody's there to click approve.

This deliberately mirrors draft_and_critique_demo.py's loop exactly (same MAX_REVISIONS
bound, same feedback-accumulation rule as build_graph.py's `_node_critic`) rather than
replacing either of them. Yes, that's the same loop logic living in three places now
(the demo script, the LangGraph node, and this function) — a real, acknowledged
duplication. The alternative was refactoring the demo script or the graph nodes to
share this, and both are already live-verified against real API output across many
rounds of debugging with no API access available in this environment to re-verify a
refactor; the risk of silently breaking working, hard-won behavior outweighed the
cost of one extra copy of a ~15-line loop. Worth revisiting later with real test
coverage backing it, not now.
"""
from __future__ import annotations

from dataclasses import dataclass

from career_copilot.config import Settings, get_settings
from career_copilot.graph.critic import critic as run_critic
from career_copilot.graph.draft_writer import draft_writer as run_draft_writer
from career_copilot.graph.gap_analysis import gap_analysis as run_gap_analysis
from career_copilot.graph.parse_jd import parse_jd as run_parse_jd
from career_copilot.graph.retrieve_evidence import EvidenceBundle, retrieve_evidence as run_retrieve_evidence
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

    for attempt in range(1, max_revisions + 2):  # 1 initial + max_revisions repairs
        draft = run_draft_writer(
            jd, gap_report, bundles, revision_feedback=feedback_history or None, settings=settings
        )
        verdict = run_critic(draft, gap_report, bundles, settings=settings)
        if verdict.passed:
            break
        revision_count = attempt
        if attempt == max_revisions + 1:
            break
        new_feedback = verdict.issues + [
            f'Claim "{c.claim_text}" is not well-grounded: {c.reason}' for c in verdict.ungrounded_claims
        ]
        for item in new_feedback:
            if item not in feedback_history:
                feedback_history.append(item)

    assert draft is not None and verdict is not None  # loop above always runs >=1 time
    return PipelineResult(
        jd=jd,
        evidence_bundles=bundles,
        gap_report=gap_report,
        draft=draft,
        critic_verdict=verdict,
        revision_count=revision_count,
    )
