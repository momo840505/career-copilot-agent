"""A skeptical second LLM pass that has to approve the draft before it's allowed
anywhere near a human. The other half of the self-correction loop -- draft_writer
proposes, critic disposes.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from career_copilot.config import Settings, get_settings
from career_copilot.graph.evidence_format import build_chunk_lookup
from career_copilot.graph.retrieve_evidence import EvidenceBundle, RetrievedChunk
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport

_SYSTEM_PROMPT = """You are a skeptical editor reviewing a drafted cover-letter pitch before \
it's allowed to reach a human for approval. Catch real problems — but do not invent problems \
that aren't there. A draft with no real issues MUST pass; padding out `issues` or \
`ungrounded_claims` to look thorough is itself a failure on your part.

Check the draft against these four rules:

1. GROUNDING — Each claim below is shown together with the actual text of its own cited \
evidence, directly underneath it — you do not need to search for or cross-reference anything. \
Just read the claim, read the evidence right below it, and judge: does that evidence mention \
the same skill, tool, or fact the claim asserts? Paraphrasing is fine; a claim does not need to \
be a verbatim quote of its source to be grounded. Only put a claim's exact text into \
`ungrounded_claims` if its own shown evidence genuinely does NOT support it (wrong topic, \
unrelated skill, or a specific number/fact absent from the source). Do not add a claim to \
`ungrounded_claims` just because the draft has some other, unrelated problem, and do not add a \
claim you have not individually checked against its own shown evidence — a default of "flag \
everything to be safe" is exactly the failure mode to avoid. For every claim you do flag, its \
`reason` must name the SPECIFIC gap between the claim and its evidence — e.g. "claim says \
'extensively used for data handling across various projects' but the evidence only lists Excel \
as one tool among several, with no detail about extent or number of projects" — not a generic \
restatement like "not supported by evidence". The writer will only see your `reason`, not your \
internal thinking, so put everything it needs to fix the sentence into that one field.

2. MISSING-SKILL HONESTY — A requirement marked "missing" must never be claimed as direct, \
hands-on experience. HOWEVER: an explicit, honestly-hedged bridge is fine and is exactly the \
behavior this tool is supposed to produce — e.g. "while I don't have direct experience with X, \
I've worked with adjacent Y" is NOT a violation. Only flag this in `issues` if the draft claims \
or implies *direct* experience with a missing requirement without that kind of disclaimer.

3. TONE — flag generic, unbacked filler ("passionate", "excited to apply", "hard-working") in \
`issues`. A normal, professional greeting/closing line is not itself a violation.

4. LENGTH — flag in `issues` only if the body is wildly off from a reasonable ~100-250 words.

Set passed=True whenever none of the above are actually violated. Otherwise set passed=False, \
and every entry in `issues` or `ungrounded_claims` must name a specific, fixable problem — not \
a vague overall impression.
"""


def _format_draft_with_inline_evidence(
    draft: CoverLetterDraft, chunk_lookup: dict[str, RetrievedChunk]
) -> str:
    """Shows each claim with the actual text of its own cited evidence directly below
    it -- does the claim<->evidence join in code instead of handing the model two
    separately-shaped lists and expecting it to cross-reference them reliably.
    Without this, the critic defaulted to flagging every single claim as
    "ungrounded" regardless of the draft's actual quality -- it wasn't failing to
    find support, it was failing to do the lookup."""
    lines = [f"Greeting: {draft.greeting}", f"\nBody:\n{draft.body}", f"\nClosing: {draft.closing}"]
    lines.append("\nClaims made, each followed by the full text of its own cited evidence:")
    for c in draft.claims:
        lines.append(f'\nCLAIM: "{c.text}"')
        if not c.evidence_chunk_ids:
            lines.append("  (no evidence_chunk_ids cited — automatically ungrounded)")
            continue
        for cid in c.evidence_chunk_ids:
            chunk = chunk_lookup.get(cid)
            if chunk is None:
                lines.append(f"  CITED [{cid}]: *** not found in retrieved evidence ***")
            else:
                lines.append(f"  CITED [{cid}] (from {chunk.doc_title}): {chunk.text}")
    return "\n".join(lines)


def _format_missing(report: GapReport) -> str:
    return "\n".join(f"- {item.requirement}" for item in report.missing) or "(none)"


def critic(
    draft: CoverLetterDraft,
    gap_report: GapReport,
    evidence_bundles: list[EvidenceBundle],
    settings: Settings | None = None,
) -> CriticVerdict:
    settings = settings or get_settings()
    # Uses its own model setting (OPENAI_CRITIC_MODEL), separate from the writer --
    # "judge whether this is good enough" is a harder call than "extract/draft".
    # config.py already had settings.critic_model; this call just wasn't passing it
    # through yet.
    llm = get_chat_model(settings, model_name=settings.critic_model)
    chunk_lookup = build_chunk_lookup(evidence_bundles)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Draft to review:\n{_format_draft_with_inline_evidence(draft, chunk_lookup)}\n\n"
                f"Requirements marked MISSING (must not be claimed):\n{_format_missing(gap_report)}"
            )
        ),
    ]
    return invoke_structured(llm, CriticVerdict, messages, node_name="critic")
