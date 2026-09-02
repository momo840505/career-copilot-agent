"""Structured output schema for the critic node (Phase 4)."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class UngroundedClaim(BaseModel):
    """A single flagged claim, paired with WHY it's flagged. A bare list of claim
    strings (the original shape) told draft_writer *that* something was wrong but not
    *what* to fix — observed live: without a reason, the writer could only reword the
    same sentence superficially and the same overclaim resurfaced attempt after attempt.
    This mirrors GapItem's (requirement, note) shape in schemas/gap.py for the same
    reason: a flag with no explanation isn't actionable feedback."""

    claim_text: str = Field(description="Exact text of the claim that isn't supported by its cited evidence.")
    reason: str = Field(
        description=(
            "Specifically what's missing or wrong — e.g. name the gap between what the claim "
            "asserts and what the cited evidence actually says (a word like 'extensively' or "
            "'expert' with no supporting detail, a fact/number absent from the source, a skill "
            "the evidence doesn't mention at all). Specific enough that the writer knows exactly "
            "what to change, not just that something is wrong."
        )
    )


class CriticVerdict(BaseModel):
    passed: bool
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found (missing-skill claims, generic filler, bad length). Empty if passed=True.",
    )
    ungrounded_claims: list[UngroundedClaim] = Field(
        default_factory=list,
        description="Claims whose cited evidence does not actually support them, each with a specific reason. Empty if passed=True.",
    )

    @model_validator(mode="after")
    def _passed_is_consistent_with_issues(self) -> "CriticVerdict":
        if self.passed and (self.issues or self.ungrounded_claims):
            raise ValueError(
                "passed=True but issues/ungrounded_claims is non-empty — that's a contradiction. "
                "If there are real problems, passed must be False."
            )
        if not self.passed and not self.issues and not self.ungrounded_claims:
            raise ValueError(
                "passed=False but no issues or ungrounded_claims were given — you must explain "
                "what's wrong so the writer can fix it."
            )
        return self
