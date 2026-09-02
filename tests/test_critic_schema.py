"""CriticVerdict schema tests — pure Pydantic logic, no API key needed."""
import pytest
from pydantic import ValidationError

from career_copilot.schemas.critic import CriticVerdict, UngroundedClaim


def test_passed_with_no_issues_is_valid():
    v = CriticVerdict(passed=True)
    assert v.issues == []
    assert v.ungrounded_claims == []


def test_passed_true_with_issues_is_rejected():
    # Self-contradictory: says it passed but also lists problems.
    with pytest.raises(ValidationError):
        CriticVerdict(passed=True, issues=["too generic"])


def test_passed_false_with_no_explanation_is_rejected():
    # Self-contradictory: says it failed but gives no reason.
    with pytest.raises(ValidationError):
        CriticVerdict(passed=False)


def test_passed_false_with_issues_is_valid():
    v = CriticVerdict(passed=False, issues=["claims LangGraph experience which is 'missing'"])
    assert v.passed is False


def test_ungrounded_claim_carries_a_reason_not_just_text():
    # The whole point of the fix: a flagged claim must come with a specific reason,
    # not just its own text, so draft_writer knows what to actually change.
    claim = UngroundedClaim(
        claim_text="My experience with Excel is solid, as I have utilized it extensively.",
        reason="Evidence only lists Excel as one tool among several, with no detail about extent of use.",
    )
    v = CriticVerdict(passed=False, ungrounded_claims=[claim])
    assert v.ungrounded_claims[0].reason != ""
    assert v.ungrounded_claims[0].claim_text.startswith("My experience")


def test_passed_false_with_only_ungrounded_claims_is_valid():
    v = CriticVerdict(
        passed=False,
        ungrounded_claims=[UngroundedClaim(claim_text="X", reason="Y is not in the evidence.")],
    )
    assert v.issues == []
    assert len(v.ungrounded_claims) == 1


def test_ungrounded_claim_requires_both_fields():
    with pytest.raises(ValidationError):
        UngroundedClaim(claim_text="X")  # missing reason
