"""Structured extraction schema for a job description (Phase 2's parse_jd node)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Seniority = Literal["intern", "entry", "mid", "senior", "lead", "manager", "unknown"]


def _clean_string_list(values: list[str]) -> list[str]:
    """Strip whitespace, drop empties, dedupe case-insensitively (keep first casing seen)."""
    seen: dict[str, str] = {}
    for v in values:
        v = v.strip()
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen[key] = v
    return list(seen.values())


class JDRequirements(BaseModel):
    """Structured extraction of a job description.

    Field `description=` text is sent to the model as part of the JSON schema — it's
    doing double duty as documentation *and* prompt engineering.
    """

    job_title: str = Field(..., min_length=1, description="The job title as stated in the JD.")
    company: str | None = Field(
        None, description="Company name if stated in the JD, otherwise null."
    )
    seniority: Seniority = Field(
        "unknown",
        description="Best guess at seniority level, inferred from the JD's title/wording.",
    )
    must_have_skills: list[str] = Field(
        ...,
        min_length=1,
        description="Skills/technologies/qualifications explicitly required (not just preferred).",
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Skills described as a plus/bonus/preferred but not required.",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Short phrases describing the core day-to-day responsibilities.",
    )
    keywords: list[str] = Field(
        ...,
        min_length=1,
        description="ATS-style keywords: tools, technologies, and domain terms worth matching on.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="1-2 sentence plain-language summary of what this role is actually about.",
    )

    @field_validator("must_have_skills", "nice_to_have_skills", "responsibilities", "keywords")
    @classmethod
    def _clean_list(cls, v: list[str]) -> list[str]:
        return _clean_string_list(v)

    @field_validator("job_title", "summary")
    @classmethod
    def _strip_str(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _must_have_wins_over_nice_to_have(self) -> "JDRequirements":
        """A skill can't be both required and merely 'nice to have'. Models do this
        occasionally (list a skill in both) — must-have wins, nice-to-have is trimmed."""
        must_lower = {s.lower() for s in self.must_have_skills}
        self.nice_to_have_skills = [
            s for s in self.nice_to_have_skills if s.lower() not in must_lower
        ]
        return self
