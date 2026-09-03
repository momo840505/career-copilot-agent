"""Phase 4, node 4: write a grounded, citation-backed cover-letter draft."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from career_copilot.config import Settings, get_settings
from career_copilot.graph.evidence_format import format_evidence_lookup
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.draft import CoverLetterDraft
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements

_SYSTEM_PROMPT = """You are drafting a short, honest cover-letter-style pitch for a real \
candidate applying to a real job. Write in first person, as her.

You will be given the job's requirements and, below, ONLY the "matched" and "partial" gap-analysis \
items together with their real supporting evidence. That's a deliberate choice, not an oversight: \
there is nothing here for a requirement the candidate doesn't have evidence for, so there is \
nothing to accidentally overclaim, bridge, or invent a citation for — just make the strongest, \
most honest case from what's actually in front of you. (The full picture, including what's \
missing, still reaches the human reviewer separately — this letter's job is to make the best \
honest case, not to enumerate every gap.)

Strict rules:
- EVERY claim in `claims` must cite at least one real chunk_id, copied EXACTLY \
character-for-character from the evidence list below. Never invent, abbreviate, or placeholder a \
chunk_id — an empty string, "_", "N/A", or anything not letter-for-letter present in the evidence \
list is NEVER valid. If you can't find a real chunk_id that actually backs a sentence, that \
sentence does not belong in `claims` at all — either drop it or rewrite it into something a real \
cited chunk does support.
- For "partial" items, honest bridging is welcome (e.g. "while I haven't directly done X, I've \
done adjacent Y") — but the claim must still cite that item's own real evidence chunk_id, same as \
any other claim.
- Keep the tone warm but factual — no generic filler ("passionate", "hard worker") unless a \
specific claim backs it. Specific and grounded beats impressive-sounding and vague.
- `body` should read as natural prose the claims are drawn from — don't just concatenate the \
claims verbatim, but every sentence that makes a factual assertion should trace to one of the \
listed claims.
- MATCH CLAIM STRENGTH TO EVIDENCE SPECIFICITY. Some evidence chunks are detailed (a project \
description with scope, numbers, outcomes) — a confident, specific claim is fine there. Other \
chunks are just a bare mention (e.g. a tool name inside a skills list, with no elaboration at \
all) — for those, the claim must stay equally bare: state plainly that you have used/are \
familiar with it, and stop there. Do NOT add strength words the evidence doesn't earn — \
"extensively", "solid", "proficient", "effectively", "various projects", "in-depth" — onto a \
claim whose only evidence is a name in a list. If you're not sure which case you're in, ask \
yourself: does the evidence describe HOW MUCH or HOW WELL I used this, or just THAT I have it? \
If it only says "that", your claim may only say "that" too — e.g. "I have used Excel as part of \
my work" is fine; "I have used Excel extensively/effectively/proficiently" is not, unless a \
DIFFERENT cited chunk actually describes the scope or outcome of that use. Rewording a claim \
that overclaims into a same-strength synonym does not fix it — the fix is removing the \
unsupported qualifier entirely, not replacing it with another one.
"""


def _format_gap_report(report: GapReport) -> str:
    """Deliberately shows ONLY matched + partial, each with its own real evidence —
    never `missing` items, and never `suggested_talking_points`. Two live failures in a
    row (see README) traced back to exactly those two fields: showing the model a
    "missing" requirement's name, or an ungrounded talking point that itself bridged a
    missing item, tempted it into writing a claim it had no real citation for — which it
    then "solved" by inventing a placeholder chunk_id. A stronger warning didn't fully
    fix that; not showing it the tempting input in the first place does. The full gap
    picture (including what's missing) still reaches the human reviewer separately —
    see human_review's payload — so nothing about honesty is lost, only removed from
    where it was actively causing harm."""
    lines = ["Matched (safe to draw on):"]
    for item in report.matched:
        lines.append(f"- {item.requirement}: {item.note} [evidence: {item.evidence_chunk_ids}]")
    lines.append("\nPartial (you may honestly bridge these, citing their own evidence):")
    for item in report.partial:
        lines.append(f"- {item.requirement}: {item.note} [evidence: {item.evidence_chunk_ids}]")
    return "\n".join(lines)


def check_claims_cite_real_evidence(draft: CoverLetterDraft, valid_ids: set[str]) -> CoverLetterDraft:
    """Same pattern as gap_analysis's citation check: the schema can't know what a
    valid chunk_id is, only the caller (who did the retrieval) knows."""
    cited = {cid for claim in draft.claims for cid in claim.evidence_chunk_ids}
    invalid = cited - valid_ids
    if invalid:
        raise ValueError(
            f"These chunk_ids were cited in claims but were never given to you as evidence: "
            f"{sorted(invalid)}. Only cite chunk_ids that appear in the evidence list above."
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

    human_content = (
        f"Job title: {jd.job_title}\n\n"
        f"Gap analysis:\n{_format_gap_report(gap_report)}\n\n"
        f"Available evidence (cite only from these chunk_ids):\n"
        f"{format_evidence_lookup(evidence_bundles)}"
    )
    if revision_feedback:
        # The FULL history, not just the latest round — otherwise a fix from attempt 1
        # (e.g. "drop the generic filler") can silently regress in attempt 3 once the
        # critic stops repeating it, because the writer has no memory of its own past.
        feedback_list = "\n".join(f"- {item}" for item in revision_feedback)
        human_content += (
            "\n\nPrevious drafts were rejected for the following reasons (this is the "
            "full history — fix every one of these, and do not reintroduce any of them "
            f"even if a more recent draft happened not to repeat it):\n{feedback_list}"
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
    return invoke_structured(
        llm,
        CoverLetterDraft,
        messages,
        validate=lambda draft: check_claims_cite_real_evidence(draft, valid_ids),
        node_name="draft_writer",
    )
