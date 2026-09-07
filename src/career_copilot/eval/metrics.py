from __future__ import annotations

from dataclasses import dataclass

from career_copilot.graph.draft_writer import (
    check_body_claim_coverage,
    check_claims_cite_real_evidence,
)
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport

_DISCLAIMER_MARKERS = (
    "don't have",
    "do not have",
    "haven't",
    "have not",
    "no direct",
    "not have direct",
    "while i",
    "lack",
    "without direct",
)


@dataclass
class MetricResult:
    name: str
    passed: bool
    detail: str = ""


def citation_validity(
    draft: CoverLetterDraft,
    evidence_bundles: list[EvidenceBundle],
) -> MetricResult:
    try:
        check_claims_cite_real_evidence(draft, all_chunk_ids(evidence_bundles))
    except ValueError as exc:
        return MetricResult("citation_validity", False, str(exc))
    return MetricResult("citation_validity", True)


def body_claim_coverage(draft: CoverLetterDraft) -> MetricResult:
    try:
        check_body_claim_coverage(draft)
    except ValueError as exc:
        return MetricResult("body_claim_coverage", False, str(exc))
    return MetricResult("body_claim_coverage", True)


def no_missing_skill_leak(
    draft: CoverLetterDraft,
    gap_report: GapReport,
) -> MetricResult:
    body_lower = draft.body.lower()
    leaks = []

    for item in gap_report.missing:
        requirement = item.requirement.strip()
        if not requirement or requirement.lower() not in body_lower:
            continue

        index = body_lower.find(requirement.lower())
        window = body_lower[
            max(0, index - 120) : index + len(requirement) + 40
        ]
        if not any(marker in window for marker in _DISCLAIMER_MARKERS):
            leaks.append(requirement)

    if leaks:
        return MetricResult(
            "no_missing_skill_leak",
            False,
            f"Missing items mentioned without disclaimer: {leaks}",
        )
    return MetricResult("no_missing_skill_leak", True)


def critic_converged(
    revision_count: int,
    critic_verdict: CriticVerdict,
    max_revisions: int,
) -> MetricResult:
    if critic_verdict.passed:
        return MetricResult(
            "critic_converged",
            True,
            f"passed after {revision_count} revision(s)",
        )
    return MetricResult(
        "critic_converged",
        False,
        f"critic still not satisfied after {revision_count}/{max_revisions} revisions",
    )
