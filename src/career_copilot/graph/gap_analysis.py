"""Compares a JD's requirements against the retrieved evidence, and tries hard to
stay honest about what's a real match versus wishful thinking."""
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


def reclassify_uncited_as_missing(report: GapReport) -> GapReport:
    """The prompt defines "matched"/"partial" as buckets that only exist because
    there's real evidence, and "missing" as the one with none -- but nothing in the
    schema enforces that. A matched/partial item with an empty evidence_chunk_ids list
    is really a "missing" item that got mis-filed. Left alone it becomes
    draft_writer's problem: _format_gap_report shows it as `[evidence: []]`,
    draft_writer still gets told to write a claim for it, and even though its prompt
    forbids inventing a chunk_id, it's reached for a placeholder like "_" instead of
    just dropping the claim.

    This used to raise ValueError and ask the model to fix it itself (reclassify as
    missing, or cite a real chunk_id) -- but on the golden eval set, for JDs whose
    only signal is an adjacent/aspirational skill ("RPA", "企業流程自動化"), the model kept
    re-asserting the same uncited classification on every retry, even after the STUCK
    note and temperature bump. It was never going to fix this one on its own.

    So this resolves it deterministically, the same way dedupe_requirements resolves
    its own conflicts, no extra LLM call needed. Moving an uncited item to 'missing'
    is always the honest move -- it doesn't invent a citation, it just accepts what
    the item's own empty evidence_chunk_ids already says. Anything with real evidence
    elsewhere is unaffected, since dedupe_requirements (which runs after this) prefers
    'partial'/'matched' over 'missing' when a requirement ends up in both.
    """
    offenders = [item for item in report.matched + report.partial if not item.evidence_chunk_ids]
    if not offenders:
        return report
    offending_reqs = {item.requirement for item in offenders}
    matched = [item for item in report.matched if item.requirement not in offending_reqs]
    partial = [item for item in report.partial if item.requirement not in offending_reqs]
    missing = list(report.missing) + offenders
    return report.model_copy(update={"matched": matched, "partial": partial, "missing": missing})


# Priority order dedupe_requirements uses to resolve a bucket conflict without going
# back to the model -- see that function's docstring for why this order specifically.
_BUCKET_KEEP_PRIORITY = ("partial", "matched", "missing")


def dedupe_requirements(report: GapReport) -> GapReport:
    """Nothing stops the model from classifying the same requirement into more than
    one bucket -- a real failure mode on the golden eval set, and a plain retry
    doesn't reliably fix it: the model would resolve the conflict on one requirement
    only to introduce a fresh one on a different requirement next attempt, for a JD
    dense enough to have several near-identical requirements to mix up. So instead of
    burning retries hoping it converges, this resolves the conflict deterministically
    -- no extra LLM call, and the result doesn't depend on which attempt came back
    cleanest.

    Resolution order, most preferred first: 'partial' > 'matched' > 'missing'.
    - A requirement the model couldn't settle on one bucket for wasn't a clear-cut
      "matched" case to begin with -- the system prompt's own tie-breaker already
      treats 'partial' as the safe default and reserves 'matched' for the obvious
      case, so when the model itself couldn't decide, 'partial' is the more honest
      of the two to keep.
    - 'missing' always loses to either, since 'missing' requires zero
      evidence_chunk_ids (enforced by GapReport's own validator) -- so anything that
      ALSO shows up as matched/partial clearly has evidence somewhere, meaning
      "missing" was just wrong for it, not a competing valid read.
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

    # `kept_bucket` only decides which *bucket* wins a cross-bucket conflict -- it
    # says nothing about a requirement the model duplicated *within* a single bucket
    # (e.g. two separate GapItems both named "SQL" inside `matched`, with no
    # cross-bucket conflict at all). The old filter below kept every item whose
    # bucket matched `kept_bucket[requirement]`, so same-bucket duplicates both
    # passed straight through -- contradicting this function's own "each
    # requirement classified exactly once" invariant. Tracking which requirements
    # have already been kept (per bucket) catches that case too, not just the
    # cross-bucket one.
    seen_in_kept_bucket: set[str] = set()
    for bucket_name, items in buckets.items():
        filtered = []
        for item in items:
            if item.requirement not in dupes:
                filtered.append(item)
                continue
            if kept_bucket[item.requirement] != bucket_name:
                continue
            if item.requirement in seen_in_kept_bucket:
                continue
            seen_in_kept_bucket.add(item.requirement)
            filtered.append(item)
        buckets[bucket_name] = filtered

    return report.model_copy(
        update={"matched": buckets["matched"], "partial": buckets["partial"], "missing": buckets["missing"]}
    )


def _validate_gap_report(report: GapReport, valid_ids: set[str]) -> GapReport:
    report = check_citations_are_real(report, valid_ids)
    report = reclassify_uncited_as_missing(report)
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
        # A denser JD means more chances for a duplicate-bucket conflict in the same
        # report, and fixing several at once is harder than fixing one -- on the
        # golden eval set, a ~10-requirement JD needed more than the default 2 repairs
        # to clear every conflict, where a shorter JD converged in 0-1. This is just
        # for this node, since its problem size scales with requirement count; the
        # others (parse_jd, draft_writer, critic) keep the default.
        max_retries=4,
        validate=lambda report: _validate_gap_report(report, valid_ids),
        node_name="gap_analysis",
    )
