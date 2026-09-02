"""JDRequirements schema tests — pure Pydantic logic, no API key needed."""
import pytest
from pydantic import ValidationError

from career_copilot.schemas.jd import JDRequirements


def _valid_kwargs(**overrides):
    base = dict(
        job_title="Data Scientist",
        must_have_skills=["Python", "SQL"],
        nice_to_have_skills=["AWS"],
        keywords=["Python", "SQL", "AWS"],
        summary="A data scientist role focused on forecasting.",
    )
    base.update(overrides)
    return base


def test_valid_input_parses():
    jd = JDRequirements(**_valid_kwargs())
    assert jd.job_title == "Data Scientist"
    assert jd.seniority == "unknown"  # default when not provided


def test_empty_must_have_skills_rejected():
    with pytest.raises(ValidationError):
        JDRequirements(**_valid_kwargs(must_have_skills=[]))


def test_empty_keywords_rejected():
    with pytest.raises(ValidationError):
        JDRequirements(**_valid_kwargs(keywords=[]))


def test_list_fields_deduped_case_insensitively():
    jd = JDRequirements(
        **_valid_kwargs(
            must_have_skills=["Python", "python", " PYTHON ", "SQL"],
        )
    )
    assert jd.must_have_skills == ["Python", "SQL"]


def test_blank_entries_stripped_from_lists():
    jd = JDRequirements(**_valid_kwargs(nice_to_have_skills=["AWS", "  ", "", "Docker"]))
    assert jd.nice_to_have_skills == ["AWS", "Docker"]


def test_must_have_wins_over_nice_to_have_on_overlap():
    jd = JDRequirements(
        **_valid_kwargs(
            must_have_skills=["Python", "SQL"],
            nice_to_have_skills=["SQL", "AWS"],  # SQL listed in both
        )
    )
    assert jd.nice_to_have_skills == ["AWS"]
    assert jd.must_have_skills == ["Python", "SQL"]


def test_job_title_and_summary_are_stripped():
    jd = JDRequirements(**_valid_kwargs(job_title="  Data Scientist  "))
    assert jd.job_title == "Data Scientist"


def test_company_defaults_to_none():
    jd = JDRequirements(**_valid_kwargs())
    assert jd.company is None
