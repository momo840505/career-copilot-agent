"""Structured output schema for the draft_writer node."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact sentence or factual clause copied verbatim from the cover-letter body. "
            "Do not paraphrase."
        ),
    )
    evidence_chunk_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "chunk_id(s) that directly support this claim. Every claim must cite at "
            "least one real supplied chunk_id."
        ),
    )

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class CoverLetterDraft(BaseModel):
    greeting: str = Field(..., min_length=1)
    body: str = Field(
        ...,
        min_length=1,
        description=(
            "The full letter body as continuous prose, 2-4 short paragraphs, first person."
        ),
    )
    closing: str = Field(..., min_length=1)
    claims: list[Claim] = Field(
        ...,
        min_length=1,
        description=(
            "Every factual/experience statement made in body, copied verbatim and backed "
            "by real evidence chunk_ids."
        ),
    )

    @field_validator("greeting", "body", "closing")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return value.strip()
