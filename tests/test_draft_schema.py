"""CoverLetterDraft schema tests — pure Pydantic logic, no API key needed."""
import pytest
from pydantic import ValidationError

from career_copilot.schemas.draft import Claim, CoverLetterDraft


def _valid_kwargs(**overrides):
    base = {
        "greeting": "Dear Hiring Manager,",
        "body": "I have hands-on experience with Python and SQL.",
        "closing": "Best regards,\nMo Wei-Ting",
        "claims": [Claim(text="I have Python experience.", evidence_chunk_ids=["skills::chunk0"])],
    }
    base.update(overrides)
    return base


def test_valid_draft_parses():
    draft = CoverLetterDraft(**_valid_kwargs())
    assert draft.claims[0].evidence_chunk_ids == ["skills::chunk0"]


def test_claim_without_evidence_rejected():
    with pytest.raises(ValidationError):
        Claim(text="I have Python experience.", evidence_chunk_ids=[])


def test_draft_without_claims_rejected():
    with pytest.raises(ValidationError):
        CoverLetterDraft(**_valid_kwargs(claims=[]))


def test_fields_are_stripped():
    draft = CoverLetterDraft(**_valid_kwargs(greeting="  Dear Hiring Manager,  "))
    assert draft.greeting == "Dear Hiring Manager,"
