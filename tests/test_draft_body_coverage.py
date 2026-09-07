import pytest

from career_copilot.graph.draft_writer import check_body_claim_coverage
from career_copilot.schemas.draft import Claim, CoverLetterDraft


def _draft(body: str, claim_texts: list[str]) -> CoverLetterDraft:
    return CoverLetterDraft(
        greeting="Dear Hiring Manager,",
        body=body,
        closing="Best regards,",
        claims=[
            Claim(text=text, evidence_chunk_ids=["doc::chunk0"])
            for text in claim_texts
        ],
    )


def test_accepts_factual_sentence_covered_by_claim():
    sentence = "I built a forecasting API with FastAPI and Docker."
    draft = _draft(sentence, [sentence])
    assert check_body_claim_coverage(draft) is draft


def test_rejects_factual_sentence_missing_from_claims():
    body = (
        "I built a forecasting API with FastAPI and Docker. "
        "I deployed the service to AWS."
    )
    draft = _draft(body, ["I built a forecasting API with FastAPI and Docker."])

    with pytest.raises(ValueError, match="Factual body sentences"):
        check_body_claim_coverage(draft)


def test_rejects_claim_not_present_in_body():
    draft = _draft(
        "I built a forecasting API with FastAPI and Docker.",
        ["I deployed the service to AWS."],
    )

    with pytest.raises(ValueError, match="Claim text must appear"):
        check_body_claim_coverage(draft)
