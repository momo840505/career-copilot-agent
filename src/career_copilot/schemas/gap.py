"""Structured output schema for the gap_analysis node (Phase 3)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GapItem(BaseModel):
    requirement: str = Field(..., min_length=1)
    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "chunk_ids (copied exactly from the evidence you were given) that support "
            "this requirement. Must be empty for 'missing' items."
        ),
    )
    note: str = Field(
        ...,
        min_length=1,
        description="One honest sentence explaining the match or gap, grounded in the cited evidence.",
    )

    @field_validator("requirement", "note")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class GapReport(BaseModel):
    matched: list[GapItem] = Field(
        default_factory=list, description="Requirements clearly and directly supported by evidence."
    )
    partial: list[GapItem] = Field(
        default_factory=list,
        description="Requirements with related but not exact/direct evidence (adjacent skill/tool).",
    )
    missing: list[GapItem] = Field(
        default_factory=list,
        description="Requirements with no relevant evidence retrieved at all. Must cite no chunk_ids.",
    )
    suggested_talking_points: list[str] = Field(
        default_factory=list,
        description="Honest ways to bridge 'partial' gaps in an interview — never invented experience.",
    )
    overall_fit_summary: str = Field(..., min_length=1)

    @field_validator("overall_fit_summary")
    @classmethod
    def _strip_summary(cls, v: str) -> str:
        return v.strip()

    @field_validator("missing")
    @classmethod
    def _missing_has_no_citations(cls, v: list[GapItem]) -> list[GapItem]:
        for item in v:
            if item.evidence_chunk_ids:
                raise ValueError(
                    f"'missing' item {item.requirement!r} cites evidence_chunk_ids "
                    f"{item.evidence_chunk_ids} — if there's evidence, it isn't missing. "
                    "Move it to 'matched' or 'partial', or drop the citation."
                )
        return v
