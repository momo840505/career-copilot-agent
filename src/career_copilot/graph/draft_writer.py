from __future__ import annotations

import re
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, SystemMessage

from career_copilot.config import Settings, get_settings
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements

_SYSTEM_PROMPT = """Write a short, factual cover letter in first person.

Use only matched and partial requirements and only the evidence listed below.

Rules:
- Every factual statement about experience, skills, projects, credentials, results, or
  numbers must be represented in claims.
- claims[].text must copy the corresponding sentence or factual clause from body
  verbatim. Do not paraphrase it.
- Every claim must cite a real chunk_id supplied below.
- Match wording strength to the evidence. Do not add unsupported qualifiers such as
  "deep expertise", "expert", "extensive", "proven", or "at scale".
- Partial matches must be described as adjacent or related experience, not as direct
  fulfillment of the requirement.
- Do not invent citations, experience, metrics, employers, tools, credentials, scale,
  seniority, or production responsibility.
- Keep the body between roughly 100 and 250 words.
"""

_FACTUAL_HINT = re.compile(
    r"\b(?:I|I've|my)\b.*\b(?:have|used|built|designed|developed|implemented|"
    r"deployed|created|managed|led|worked|trained|analyzed|analysed|processed|"
    r"achieved|improved|wrote|tested|experience|degree|project)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9+#./-]+")
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|haven't|hasn't|don't|doesn't|didn't)\b|"
    r"\b(?:have|has|do|does|did)\s+not\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "as", "at", "for", "from", "i", "in", "into",
    "my", "of", "on", "the", "to", "using", "with",
}


def _normalize(value: str) -> str:
    return " ".join(value.split()).strip().casefold()


def _split_sentences(body: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if sentence.strip()
    ]


def _content_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(value)
        if token.casefold() not in _STOPWORDS
    }


def _same_negation_polarity(left: str, right: str) -> bool:
    return bool(_NEGATION_RE.search(left)) == bool(_NEGATION_RE.search(right))


def _alignment_score(claim: str, sentence: str) -> float:
    claim_norm = _normalize(claim)
    sentence_norm = _normalize(sentence)
    sequence = SequenceMatcher(None, claim_norm, sentence_norm).ratio()

    claim_tokens = _content_tokens(claim)
    sentence_tokens = _content_tokens(sentence)
    if not claim_tokens or not sentence_tokens:
        return sequence

    overlap = len(claim_tokens & sentence_tokens)
    containment = overlap / min(len(claim_tokens), len(sentence_tokens))
    return max(sequence, containment)


def align_claims_to_body(draft: CoverLetterDraft) -> CoverLetterDraft:
    sentences = _split_sentences(draft.body)
    if not sentences:
        return draft

    updated = []
    changed = False

    for claim in draft.claims:
        claim_norm = _normalize(claim.text)
        if any(
            claim_norm in _normalize(sentence) or _normalize(sentence) in claim_norm
            for sentence in sentences
        ):
            updated.append(claim)
            continue

        candidates = [
            sentence
            for sentence in sentences
            if _same_negation_polarity(claim.text, sentence)
        ]
        if not candidates:
            updated.append(claim)
            continue

        best = max(candidates, key=lambda sentence: _alignment_score(claim.text, sentence))
        if _alignment_score(claim.text, best) >= 0.68:
            updated.append(claim.model_copy(update={"text": best}))
            changed = True
        else:
            updated.append(claim)

    if not changed:
        return draft
    return draft.model_copy(update={"claims": updated})


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


def _allowed_evidence_ids(report: GapReport) -> set[str]:
    return {
        chunk_id
        for item in report.matched + report.partial
        for chunk_id in item.evidence_chunk_ids
    }


def _format_allowed_evidence(
    bundles: list[EvidenceBundle],
    allowed_ids: set[str],
) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for bundle in bundles:
        for chunk in bundle.chunks:
            if chunk.chunk_id not in allowed_ids or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            lines.append(
                f"- chunk_id={chunk.chunk_id} (from {chunk.doc_title}): {chunk.text}"
            )
    return "\n".join(lines) or "(no citable evidence)"


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
            f"Unknown or disallowed evidence chunk_ids in draft claims: {sorted(invalid)}"
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

    uncovered = []
    for sentence in _split_sentences(draft.body):
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


def _validate_draft(
    draft: CoverLetterDraft,
    valid_ids: set[str],
) -> CoverLetterDraft:
    draft = check_claims_cite_real_evidence(draft, valid_ids)
    draft = align_claims_to_body(draft)
    return check_body_claim_coverage(draft)


def draft_writer(
    jd: JDRequirements,
    gap_report: GapReport,
    evidence_bundles: list[EvidenceBundle],
    revision_feedback: list[str] | None = None,
    settings: Settings | None = None,
) -> CoverLetterDraft:
    settings = settings or get_settings()
    llm = get_chat_model(settings, temperature=0.2)

    allowed_ids = _allowed_evidence_ids(gap_report)
    content = (
        f"Job title: {jd.job_title}\n\n"
        f"Gap analysis:\n{_format_gap_report(gap_report)}\n\n"
        "Citable evidence:\n"
        f"{_format_allowed_evidence(evidence_bundles, allowed_ids)}"
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
        max_retries=4,
        validate=lambda draft: _validate_draft(draft, allowed_ids),
        node_name="draft_writer",
    )
