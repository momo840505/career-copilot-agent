"""Structured output schema for the draft_writer node (Phase 4)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    text: str = Field(
        ..., min_length=1, description="A single factual/experience claim made in the letter, first person."
    )
    evidence_chunk_ids: list[str] = Field(
        ...,
        min_length=1,
        description="chunk_id(s) that directly support this claim. Every claim must cite at least one — no exceptions.",
    )

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class CoverLetterDraft(BaseModel):
    greeting: str = Field(..., min_length=1)
    body: str = Field(
        ...,
        min_length=1,
        description="The full letter body as continuous prose, 2-4 short paragraphs, first person.",
    )
    closing: str = Field(..., min_length=1)
    claims: list[Claim] = Field(
        ...,
        min_length=1,
        description="Every factual/experience claim made in `body`, each backed by real chunk_id(s).",
    )

    @field_validator("greeting", "body", "closing")
    @classmethod
    def _strip_fields(cls, v: str) -> str:
        return v.strip()
