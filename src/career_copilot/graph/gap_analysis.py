from __future__ import annotations

from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

from career_copilot.config import Settings, get_settings
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.gap import GapReport
from career_copilot.schemas.jd import JDRequirements

_SYSTEM_PROMPT = """Compare the candidate evidence with each job requirement.

Rules:
- Copy each requirement text exactly as provided.
- Classify every requirement exactly once as matched, partial, or missing.
- matched: the evidence directly supports the requirement.
- partial: the evidence is relevant but does not directly support the full requirement.
- missing: no relevant evidence was retrieved.
- matched and partial items must cite only chunk_ids listed under that same requirement.
- missing items must not cite evidence.
- Do not add requirements that are not in the input.
- Keep notes factual and specific.
- Suggested talking points may bridge partial gaps, but must not invent experience.
"""


def _key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _format_evidence(bundles: list[EvidenceBundle]) -> str:
    lines: list[str] = []
    for bundle in bundles:
        lines.append(f"\n### Requirement: {bundle.requirement}")
        if not bundle.chunks:
            lines.append("(no evidence retrieved)")
            continue
        for chunk in bundle.chunks:
            lines.append(
                f"- chunk_id={chunk.chunk_id} (from {chunk.doc_title}): {chunk.text}"
            )
    return "\n".join(lines)


def check_citations_are_real(report: GapReport, valid_ids: set[str]) -> GapReport:
    cited = {
        cid
        for item in report.matched + report.partial
        for cid in item.evidence_chunk_ids
    }
    invalid = cited - valid_ids
    if invalid:
        raise ValueError(
            f"Unknown evidence chunk_ids: {sorted(invalid)}. "
            "Use only chunk_ids supplied in the evidence."
        )
    return report


def reclassify_uncited_as_missing(report: GapReport) -> GapReport:
    offenders = [
        item for item in report.matched + report.partial if not item.evidence_chunk_ids
    ]
    if not offenders:
        return report

    names = {_key(item.requirement) for item in offenders}
    matched = [item for item in report.matched if _key(item.requirement) not in names]
    partial = [item for item in report.partial if _key(item.requirement) not in names]
    missing = list(report.missing) + offenders
    return report.model_copy(
        update={"matched": matched, "partial": partial, "missing": missing}
    )


_BUCKET_KEEP_PRIORITY = ("partial", "matched", "missing")


def dedupe_requirements(report: GapReport) -> GapReport:
    buckets = {
        "matched": list(report.matched),
        "partial": list(report.partial),
        "missing": list(report.missing),
    }
    counts = Counter(
        _key(item.requirement) for items in buckets.values() for item in items
    )
    duplicates = {name for name, count in counts.items() if count > 1}
    if not duplicates:
        return report

    keep_bucket: dict[str, str] = {}
    for bucket_name in _BUCKET_KEEP_PRIORITY:
        for item in buckets[bucket_name]:
            name = _key(item.requirement)
            if name in duplicates and name not in keep_bucket:
                keep_bucket[name] = bucket_name

    seen: set[str] = set()
    for bucket_name, items in buckets.items():
        filtered = []
        for item in items:
            name = _key(item.requirement)
            if name not in duplicates:
                filtered.append(item)
                continue
            if keep_bucket[name] != bucket_name or name in seen:
                continue
            seen.add(name)
            filtered.append(item)
        buckets[bucket_name] = filtered

    return report.model_copy(update=buckets)


def check_requirement_contract(
    report: GapReport,
    bundles: list[EvidenceBundle],
    expected_requirements: list[str],
) -> GapReport:
    expected = {_key(requirement): requirement for requirement in expected_requirements}
    classified = {
        _key(item.requirement): item
        for item in report.matched + report.partial + report.missing
    }

    unclassified = sorted(set(expected) - set(classified))
    invented = sorted(set(classified) - set(expected))
    if unclassified or invented:
        errors = []
        if unclassified:
            errors.append(
                "requirements not classified: "
                + ", ".join(repr(expected[name]) for name in unclassified)
            )
        if invented:
            errors.append(
                "requirements not present in the JD: "
                + ", ".join(repr(classified[name].requirement) for name in invented)
            )
        raise ValueError("; ".join(errors))

    allowed_by_requirement = {
        _key(bundle.requirement): {chunk.chunk_id for chunk in bundle.chunks}
        for bundle in bundles
    }
    for item in report.matched + report.partial:
        allowed = allowed_by_requirement.get(_key(item.requirement), set())
        wrong = set(item.evidence_chunk_ids) - allowed
        if wrong:
            raise ValueError(
                f"{item.requirement!r} cites evidence retrieved for another requirement: "
                f"{sorted(wrong)}"
            )

    return report


def _validate_gap_report(
    report: GapReport,
    bundles: list[EvidenceBundle],
    expected_requirements: list[str],
) -> GapReport:
    report = check_citations_are_real(report, all_chunk_ids(bundles))
    report = reclassify_uncited_as_missing(report)
    report = dedupe_requirements(report)
    return check_requirement_contract(report, bundles, expected_requirements)


def gap_analysis(
    jd: JDRequirements,
    evidence_bundles: list[EvidenceBundle],
    settings: Settings | None = None,
) -> GapReport:
    settings = settings or get_settings()
    llm = get_chat_model(settings)
    expected = jd.must_have_skills + jd.nice_to_have_skills

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
        max_retries=4,
        validate=lambda report: _validate_gap_report(
            report, evidence_bundles, expected
        ),
        node_name="gap_analysis",
    )
