"""Regression test for a revision_count off-by-one in graph/pipeline.py.

run_pipeline's non-interactive draft/critic loop (used by scripts/run_evals.py and
the API's /draft route) mislabeled "which attempt number is this" as "how many
revisions have been performed". That's correct on every path that eventually
converges, but wrong by exactly 1 whenever the retry budget is fully exhausted
without ever passing: a run capped at max_revisions=2 would report
revision_count=3, making eval/metrics.py's critic_converged check print a
confusing "3/2 revisions" for a run that never actually exceeded its own budget.

These tests drive run_pipeline's loop directly -- parse_jd/retrieve_evidence/
gap_analysis/draft_writer are monkeypatched to fixed fakes, and only the critic's
pass/fail sequence varies per test -- across every revision_count-affecting
outcome. No API key needed, same spirit as test_invoke_structured_retry.py's
fake-LLM harness.
"""
from __future__ import annotations

import career_copilot.graph.pipeline as pipeline_module
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements


def _fake_jd() -> JDRequirements:
    return JDRequirements(
        job_title="Data Analyst",
        must_have_skills=["SQL"],
        keywords=["SQL"],
        summary="A data analyst role.",
    )


def _fake_gap_report() -> GapReport:
    return GapReport(overall_fit_summary="Good fit.")


def _fake_draft() -> CoverLetterDraft:
    return CoverLetterDraft(
        greeting="Dear Hiring Manager,",
        body="I have relevant experience.",
        closing="Sincerely,",
        claims=[Claim(text="I know SQL.", evidence_chunk_ids=["skills::chunk0"])],
    )


def _verdict(passed: bool) -> CriticVerdict:
    return CriticVerdict(passed=passed, issues=[] if passed else ["too generic"])


def _patch_upstream(monkeypatch):
    monkeypatch.setattr(pipeline_module, "run_parse_jd", lambda jd_text, settings=None: _fake_jd())
    monkeypatch.setattr(pipeline_module, "run_retrieve_evidence", lambda jd, settings=None: [])
    monkeypatch.setattr(
        pipeline_module, "run_gap_analysis", lambda jd, bundles, settings=None: _fake_gap_report()
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_draft_writer",
        lambda jd, gap_report, bundles, revision_feedback=None, settings=None: _fake_draft(),
    )


def _run_with_pass_sequence(monkeypatch, pass_sequence: list[bool]):
    """pass_sequence[i] is the critic's verdict for the (i+1)-th attempt."""
    _patch_upstream(monkeypatch)
    calls = {"n": 0}

    def fake_critic(draft, gap_report, bundles, settings=None):
        i = calls["n"]
        calls["n"] += 1
        return _verdict(pass_sequence[i])

    monkeypatch.setattr(pipeline_module, "run_critic", fake_critic)
    return pipeline_module.run_pipeline("irrelevant jd text", max_revisions=2)


def test_passes_immediately_reports_zero_revisions(monkeypatch):
    result = _run_with_pass_sequence(monkeypatch, [True])
    assert result.critic_verdict.passed is True
    assert result.revision_count == 0


def test_passes_after_one_revision(monkeypatch):
    result = _run_with_pass_sequence(monkeypatch, [False, True])
    assert result.critic_verdict.passed is True
    assert result.revision_count == 1


def test_passes_on_the_last_attempt_the_budget_allows(monkeypatch):
    # 3rd attempt (= 2nd revision, the last one max_revisions=2 allows) passes.
    result = _run_with_pass_sequence(monkeypatch, [False, False, True])
    assert result.critic_verdict.passed is True
    assert result.revision_count == 2


def test_never_passes_reports_exactly_max_revisions_not_one_more(monkeypatch):
    # The regression this file guards against: before the fix, this reported
    # revision_count == 3 (max_revisions + 1) even though only 2 revisions -- the
    # configured budget -- were ever attempted.
    result = _run_with_pass_sequence(monkeypatch, [False, False, False])
    assert result.critic_verdict.passed is False
    assert result.revision_count == 2
