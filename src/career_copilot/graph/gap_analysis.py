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


def check_no_duplicate_requirements(report: GapReport) -> GapReport:
    """Same principle as check_citations_are_real: a check that needs to see the
    *whole* report at once (not just one field) has to live outside the schema.

    Nothing stops the model from classifying the same requirement as both "matched"
    and "missing" — a genuinely observed failure mode, not a hypothetical one — so
    this catches it and forces a repair rather than silently shipping a
    self-contradictory report.
    """
    all_reqs = [item.requirement for item in report.matched + report.partial + report.missing]
    dupes = {req: count for req, count in Counter(all_reqs).items() if count > 1}
    if dupes:
        raise ValueError(
            f"These requirements were classified into more than one bucket "
            f"(matched/partial/missing): {dupes}. Each requirement must appear in "
            f"exactly one bucket — pick the single best-fitting one and remove it "
            f"from the others."
        )
    return report


def _validate_gap_report(report: GapReport, valid_ids: set[str]) -> GapReport:
    report = check_citations_are_real(report, valid_ids)
    report = check_no_duplicate_requirements(report)
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
