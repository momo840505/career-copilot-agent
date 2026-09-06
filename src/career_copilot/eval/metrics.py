"""Objective, code-based metrics for a completed pipeline run.

Deliberately NOT LLM-as-judge -- these are things a set-membership check or a string
search can verify exactly, so they're deterministic, free, and reproducible, unlike
the groundedness judge in groundedness_judge.py. That's also why these are the metrics
treated as hard CI gates (a failing run fails the build) while groundedness is treated
as an informational score to watch, not a gate: a metric a human would have to
double-check by hand isn't trustworthy enough to block a merge on by itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from career_copilot.graph.draft_writer import check_claims_cite_real_evidence
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport

# Phrases that plausibly signal "I'm disclaiming this, not claiming it" near a mention
# of a missing requirement. Deliberately over-inclusive (a false negative here just
# means no_missing_skill_leak misses a real leak, which the human reviewer would still
# catch) rather than under-inclusive (which would flag honest disclaimers as failures
# and make the metric useless as a gate).
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


def citation_validity(draft: CoverLetterDraft, evidence_bundles: list[EvidenceBundle]) -> MetricResult:
    """Every claim cites only real, retrieved chunk_ids. draft_writer already
    self-enforces this via invoke_structured's validate=, so a failure here would mean
    that safety net itself regressed — this is an independent, outside-the-loop
    confirmation, not a duplicate check for its own sake."""
    valid_ids = all_chunk_ids(evidence_bundles)
    try:
        check_claims_cite_real_evidence(draft, valid_ids)
    except ValueError as e:
        return MetricResult("citation_validity", False, str(e))
    return MetricResult("citation_validity", True)


def no_missing_skill_leak(draft: CoverLetterDraft, gap_report: GapReport) -> MetricResult:
    """Does the letter body mention a 'missing' requirement's exact text without an
    obvious disclaimer nearby? Intentionally a blunt substring check, not an LLM judge:
    it can't catch a paraphrased overclaim, but what it DOES catch, it catches with
    zero false positives on an honest draft and zero API cost — which is what makes it
    safe to use as a hard gate rather than an advisory score. (draft_writer is no
    longer even shown the missing list, so this should always pass -- this metric
    exists to catch a regression if that ever changes.)"""
    body_lower = draft.body.lower()
    leaks = []
    for item in gap_report.missing:
        req = item.requirement.strip()
        if not req or req.lower() not in body_lower:
            continue
        idx = body_lower.find(req.lower())
        window = body_lower[max(0, idx - 120) : idx + len(req) + 40]
        if not any(marker in window for marker in _DISCLAIMER_MARKERS):
            leaks.append(req)
    if leaks:
        return MetricResult("no_missing_skill_leak", False, f"Missing items mentioned without disclaimer: {leaks}")
    return MetricResult("no_missing_skill_leak", True)


def critic_converged(revision_count: int, critic_verdict: CriticVerdict, max_revisions: int) -> MetricResult:
    if critic_verdict.passed:
        return MetricResult("critic_converged", True, f"passed after {revision_count} revision(s)")
    return MetricResult(
        "critic_converged",
        False,
        f"critic still not satisfied after {revision_count}/{max_revisions} revisions",
    )
