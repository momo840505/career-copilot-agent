"""Phase 3, node 3: compare JD requirements against retrieved evidence, honestly."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from collections import Counter

from career_copilot.config import Settings, get_settings
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements

_SYSTEM_PROMPT = """You are a rigorous, honest career coach comparing a candidate's real \
background against a job description's requirements.

You will be given the job's must-have and nice-to-have requirements, and for each one, a \
small set of retrieved evidence chunks from the candidate's actual resume/portfolio (each \
with a chunk_id).

Strict rules:
- Only cite chunk_ids that were actually given to you below. Never invent one.
- Classify each requirement: "matched" (evidence clearly and directly supports it), \
"partial" (related but not a direct match — an adjacent tool or domain), or "missing" (no \
relevant evidence was retrieved at all).
- A "missing" requirement must NOT cite any evidence_chunk_ids.
- Each requirement must be classified into exactly ONE bucket — matched, partial, OR \
missing, never more than one. Do not list the same requirement twice.
- TIE-BREAKER for a narrow sub-skill under a broader tool/domain you have general evidence \
for (e.g. requirement = "pivot tables", evidence = general "Excel" experience that never \
specifically names pivot tables): this is a "partial" match, not "matched" AND something \
else. Use this exact rule, in order: (1) if the evidence explicitly names the specific \
sub-skill (or something functionally identical), classify "matched"; (2) otherwise, if the \
evidence only supports the broader tool/domain without naming the specific sub-skill, \
classify "partial" — evidence of the general tool is real signal, so this is never "missing"; \
(3) only classify "missing" when there is no relevant evidence for the broader domain either. \
Apply this rule once, silently, and commit to its answer — do not place the same requirement \
in more than one bucket while you weigh the two readings against each other.
- Do not inflate or spin. If the evidence is weak, say "partial", not "matched". Being \
honest about real gaps is the entire point of this tool — the candidate needs to know where \
she actually stands, not be flattered.
- suggested_talking_points should honestly bridge "partial" gaps (e.g. "no direct framework \
X experience, but built a comparable pipeline with Y") — never invent experience that isn't \
in the evidence.
"""


def _format_evidence(bundles: list[EvidenceBundle]) -> str:
    lines: list[str] = []
    for b in bundles:
        lines.append(f"\n### Requirement: {b.requirement}")
        if not b.chunks:
            lines.append("(no evidence retrieved)")
            continue
        for c in b.chunks:
            lines.append(f"- chunk_id={c.chunk_id} (from {c.doc_title}): {c.text}")
    return "\n".join(lines)


def check_citations_are_real(report: GapReport, valid_ids: set[str]) -> GapReport:
    """The schema can't know what a valid chunk_id is — only the caller (who did the
    retrieval) knows. This is exactly the kind of check `invoke_structured`'s
    `validate=` hook exists for: catch a hallucinated citation and force a repair."""
    cited = {cid for item in (report.matched + report.partial) for cid in item.evidence_chunk_ids}
    invalid = cited - valid_ids
    if invalid:
        raise ValueError(
            f"These chunk_ids were cited but were never given to you as evidence: "
            f"{sorted(invalid)}. Only cite chunk_ids that appear in the evidence list above."
        )
    return report


def check_matched_and_partial_have_evidence(report: GapReport) -> GapReport:
    """The system prompt defines "matched"/"partial" as buckets that exist *because*
    there's real evidence, and "missing" as the one with none — but nothing in the
    schema enforced that. A matched/partial item with an empty evidence_chunk_ids list
    is a "missing" item mis-filed, and left uncaught it becomes draft_writer's problem
    instead: _format_gap_report shows it as `[evidence: []]`, draft_writer is told to
    write a claim for it anyway, and — even though its own prompt explicitly forbids
    ever inventing a chunk_id — under that pressure it has been observed reaching for a
    placeholder like "_" rather than dropping the claim (see draft_writer.py's module
    docstring, and the golden-eval failure that motivated this check). Catching the
    empty-evidence misclassification here, at the source, is more reliable than asking
    a downstream node to resist writing about something it was handed as if it were
    legitimate.
    """
    offenders = [item.requirement for item in report.matched + report.partial if not item.evidence_chunk_ids]
    if offenders:
        raise ValueError(
            f"These requirements are classified as matched/partial but cite no "
            f"evidence_chunk_ids: {offenders}. A requirement with no real evidence "
            f"chunk to cite isn't a match at all — reclassify it as 'missing' (with no "
            f"citations and a note explaining nothing relevant was retrieved), or if "
            f"there genuinely is supporting evidence, cite the specific chunk_id(s) "
            f"that show it."
        )
    return report


# Preference order used by dedupe_requirements when the SAME bucket-conflict is
# resolved without going back to the model — see that function's docstring for why
# this order, specifically, is the honest choice rather than an arbitrary one.
_BUCKET_KEEP_PRIORITY = ("partial", "matched", "missing")


def dedupe_requirements(report: GapReport) -> GapReport:
    """Nothing stops the model from classifying the same requirement into more than
    one bucket — a genuinely observed failure mode (see the golden eval set), and one
    where a plain retry doesn't reliably converge: watched live, the model kept
    resolving the conflict on one requirement only to introduce a fresh one on a
    *different* requirement, attempt after attempt, for a JD dense enough to have
    several near-identical requirements to get confused between. Rather than keep
    spending retries hoping the model eventually lands on a single self-consistent
    report, this resolves the conflict deterministically and returns a corrected
    report — no extra LLM call needed, and the result doesn't depend on which attempt
    happened to come back cleanest.

    Resolution order, most-preferred bucket first: 'partial' > 'matched' > 'missing'.
    - A requirement the model couldn't commit to a single bucket for was, by
      construction, not a case where "matched" was unambiguous — the gap_analysis
      system prompt's own tie-breaker rule already treats 'partial' as the safe
      default and reserves 'matched' for the clear-cut case, so when the model itself
      couldn't decide, 'partial' is the more honest of the two to keep.
    - 'missing' always loses to either: 'missing' requires zero evidence_chunk_ids
      (enforced separately, see GapReport's own validator), so a requirement that ALSO
      appears as matched/partial has real evidence somewhere — "missing" was simply
      wrong for it, not a competing valid judgment.
    """
    buckets = {"matched": list(report.matched), "partial": list(report.partial), "missing": list(report.missing)}
    counts = Counter(item.requirement for items in buckets.values() for item in items)
    dupes = {req for req, count in counts.items() if count > 1}
    if not dupes:
        return report

    kept_bucket: dict[str, str] = {}
    for bucket_name in _BUCKET_KEEP_PRIORITY:
        for item in buckets[bucket_name]:
            if item.requirement in dupes and item.requirement not in kept_bucket:
                kept_bucket[item.requirement] = bucket_name

    for bucket_name, items in buckets.items():
        buckets[bucket_name] = [
            item for item in items if item.requirement not in dupes or kept_bucket[item.requirement] == bucket_name
        ]

    return report.model_copy(
        update={"matched": buckets["matched"], "partial": buckets["partial"], "missing": buckets["missing"]}
    )


def _validate_gap_report(report: GapReport, valid_ids: set[str]) -> GapReport:
    report = check_citations_are_real(report, valid_ids)
    report = check_matched_and_partial_have_evidence(report)
    report = dedupe_requirements(report)
    return report


def gap_analysis(
    jd: JDRequirements,
    evidence_bundles: list[EvidenceBundle],
    settings: Settings | None = None,
) -> GapReport:
    settings = settings or get_settings()
    llm = get_chat_model(settings)
    valid_ids = all_chunk_ids(evidence_bundles)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Job title: {jd.job_title}\n"
                f"Must-have requirements: {jd.must_have_skills}\n"
                f"Nice-to-have requirements: {jd.nice_to_have_skills}\n"
                f"\nRetrieved evidence per requirement:\n{_format_evidence(evidence_bundles)}"
            )
        ),
    ]
    return invoke_structured(
        llm,
        GapReport,
        messages,
        # A denser JD (more requirements) means more independent chances for a
        # duplicate-bucket conflict in the SAME report, and fixing several at once is a
        # harder combinatorial problem than fixing one — observed live on the golden
        # eval set: a JD with ~10 requirements needed more than the default 2 repairs to
        # shake every conflict out, where a shorter JD converged in 0-1. Every other node
        # (parse_jd, draft_writer, critic) keeps the invoke_structured default; this is a
        # deliberate, narrow widening for the one node whose problem size scales with the
        # JD's requirement count, not a global "just retry more" change.
        max_retries=4,
        validate=lambda report: _validate_gap_report(report, valid_ids),
    )
