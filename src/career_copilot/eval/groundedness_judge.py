"""A separate LLM-as-judge pass, used only for evals -- not part of the production
critic loop in graph/critic.py.

This is where the "LLM-as-judge noise" limitation flagged in the README's engineering
notes (the same sentence's verdict could flip between calls) gets addressed directly,
instead of just noted: rather than trusting one judge call,
ask the same question `n_votes` times and look at the SPREAD, not just the average. A
tight cluster of scores is a signal the judge's opinion is real and reproducible; a
wide spread is a signal today's single score would have been noise, not a genuine
quality signal — worth surfacing to whoever's reading the eval report, not hiding
behind one clean-looking number.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from career_copilot.config import Settings, get_settings
from career_copilot.graph.evidence_format import build_chunk_lookup
from career_copilot.graph.retrieve_evidence import EvidenceBundle, RetrievedChunk
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.draft import CoverLetterDraft


class GroundednessScore(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0 = claims mostly overclaim or aren't supported by their evidence; 1.0 = every claim is fully and specifically grounded in its cited evidence.",
    )
    rationale: str = Field(..., min_length=1, description="One or two sentences justifying the score.")


@dataclass
class GroundednessResult:
    votes: list[float]
    median_score: float
    spread: float  # max - min across votes: a cheap, visible noise signal
    rationales: list[str]


_JUDGE_PROMPT = """Score how well this cover-letter draft's claims are grounded in their own \
cited evidence, on a scale from 0.0 (mostly overclaiming or unsupported) to 1.0 (every claim is \
fully and specifically supported by its cited evidence). Judge ONLY groundedness — not tone, \
length, or writing quality; a well-grounded but plainly-written draft should score as high as an \
eloquent one. Each claim below is shown together with the actual text of its own cited evidence."""


def _format_draft_with_evidence(draft: CoverLetterDraft, chunk_lookup: dict[str, RetrievedChunk]) -> str:
    lines = [f"Body:\n{draft.body}", "\nClaims, each with its own cited evidence:"]
    for c in draft.claims:
        lines.append(f'\nCLAIM: "{c.text}"')
        for cid in c.evidence_chunk_ids:
            chunk = chunk_lookup.get(cid)
            lines.append(f"  EVIDENCE [{cid}]: {chunk.text if chunk else '(not found in retrieved evidence)'}")
    return "\n".join(lines)


def judge_groundedness(
    draft: CoverLetterDraft,
    evidence_bundles: list[EvidenceBundle],
    n_votes: int = 3,
    settings: Settings | None = None,
) -> GroundednessResult:
    settings = settings or get_settings()
    # A non-zero temperature is deliberate here (unlike the production critic, which
    # runs at the default 0.0): the whole point of voting n_votes times is to observe
    # the judge's natural variance, which a temperature of 0 would mostly suppress.
    llm = get_chat_model(settings, model_name=settings.critic_model, temperature=0.5)
    chunk_lookup = build_chunk_lookup(evidence_bundles)
    content = _format_draft_with_evidence(draft, chunk_lookup)
    messages = [SystemMessage(content=_JUDGE_PROMPT), HumanMessage(content=content)]

    votes: list[float] = []
    rationales: list[str] = []
    for _ in range(n_votes):
        result = invoke_structured(llm, GroundednessScore, messages, node_name="groundedness_judge")
        votes.append(result.score)
        rationales.append(result.rationale)

    return GroundednessResult(
        votes=votes,
        median_score=statistics.median(votes),
        spread=max(votes) - min(votes),
        rationales=rationales,
    )
