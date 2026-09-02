"""End-to-end parse_jd smoke test — needs a real OpenAI API key (it calls the chat
completions API), so it's skipped automatically when no key is configured. Run
explicitly with:

    pytest tests/test_parse_jd_smoke.py -m requires_api -v
"""
import os

import pytest

from career_copilot.graph.parse_jd import parse_jd

pytestmark = pytest.mark.requires_api

needs_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set — skipping live API test"
)

SAMPLE_JD = """
Data Scientist — Acme Analytics

We're looking for a Data Scientist to join our growth team.

Requirements:
- 3+ years of experience with Python and SQL
- Experience deploying models to production (Docker, cloud platforms like AWS or GCP)
- Strong communication skills, comfortable presenting to non-technical stakeholders

Nice to have:
- Experience with time-series forecasting
- Familiarity with Tableau or Power BI
"""


@needs_key
def test_parses_sample_jd_into_valid_schema():
    result = parse_jd(SAMPLE_JD)

    assert "data scientist" in result.job_title.lower()
    must_have_lower = {s.lower() for s in result.must_have_skills}
    assert must_have_lower & {"python", "sql"}
    assert len(result.keywords) > 0
    assert len(result.summary) > 0
