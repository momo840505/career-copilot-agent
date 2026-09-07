from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from career_copilot.config import Settings, get_settings
from career_copilot.graph.evidence_format import format_evidence_lookup
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements

_SYSTEM_PROMPT = """Write a short, factual cover letter in first person.

Use only the matched and partial requirements and the evidence supplied below.

Rules:
- Every factual statement about experience, skills, projects, credentials, results, or
  numbers must be represented in claims.
- Claim text should be the sentence, or the factual clause, used in the body.
- Every claim must cite a real chunk_id supplied below.
- Match the strength of the wording to the evidence. Do not add unsupported qualifiers.
- Partial matches may be described as adjacent experience, but not as direct experience.
- Do not invent citations, experience, metrics, employers, tools, or credentials.
- Keep the body between roughly 100 and 250 words.
"""

_FACTUAL_HINT = re.compile(
    r"\b(?:I|I've|my)\b.*\b(?:have|used|built|designed|developed|implemented|"
    r"deployed|created|managed|led|worked|trained|analyzed|analysed|processed|"
    r"achieved|improved|wrote|tested|experience|degree|project)\b",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return " ".join(value.split()).strip().casefold()


def _format_gap_report(report: GapReport) -> str:
    lines = ["Matched:"]
    for item in report.matched:
        lines.append(
            f"- {item.requirement}: {item.note} [evidence: {item.evidence_chunk_ids}]"
        )

    lines.append("\nPartial:")
    for item in report.partial:
        lines.append(
            f"- {item.requirement}: {item.note} [evidence: {item.evidence_chunk_ids}]"
        )
    return "\n".join(lines)


def check_claims_cite_real_evidence(
    draft: CoverLetterDraft,
    valid_ids: set[str],
) -> CoverLetterDraft:
    cited = {
        chunk_id
        for claim in draft.claims
        for chunk_id in claim.evidence_chunk_ids
    }
    invalid = cited - valid_ids
    if invalid:
        raise ValueError(
            f"Unknown evidence chunk_ids in draft claims: {sorted(invalid)}"
        )
    return draft


def check_body_claim_coverage(draft: CoverLetterDraft) -> CoverLetterDraft:
    body = _normalize(draft.body)
    claims = [_normalize(claim.text) for claim in draft.claims]

    missing_from_body = [
        claim.text
        for claim, normalized in zip(draft.claims, claims)
        if normalized not in body
    ]
    if missing_from_body:
        raise ValueError(
            "Claim text must appear in the cover-letter body: "
            f"{missing_from_body}"
        )

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", draft.body)
        if sentence.strip()
    ]
    uncovered = []
    for sentence in sentences:
        if not _FACTUAL_HINT.search(sentence):
            continue
        normalized = _normalize(sentence)
        if not any(
            normalized in claim or claim in normalized
            for claim in claims
        ):
            uncovered.append(sentence)

    if uncovered:
        raise ValueError(
            "Factual body sentences must be represented in claims: "
            f"{uncovered}"
        )
    return draft


def draft_writer(
    jd: JDRequirements,
    gap_report: GapReport,
    evidence_bundles: list[EvidenceBundle],
    revision_feedback: list[str] | None = None,
    settings: Settings | None = None,
) -> CoverLetterDraft:
    settings = settings or get_settings()
    llm = get_chat_model(settings, temperature=0.3)
    valid_ids = all_chunk_ids(evidence_bundles)

    content = (
        f"Job title: {jd.job_title}\n\n"
        f"Gap analysis:\n{_format_gap_report(gap_report)}\n\n"
        "Available evidence:\n"
        f"{format_evidence_lookup(evidence_bundles)}"
    )
    if revision_feedback:
        feedback = "\n".join(f"- {item}" for item in revision_feedback)
        content += f"\n\nRevision feedback:\n{feedback}"

    return invoke_structured(
        llm,
        CoverLetterDraft,
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ],
        validate=lambda draft: check_body_claim_coverage(
            check_claims_cite_real_evidence(draft, valid_ids)
        ),
        node_name="draft_writer",
    )
