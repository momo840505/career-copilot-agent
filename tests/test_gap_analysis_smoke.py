"""End-to-end smoke test for the full parse_jd -> retrieve_evidence -> gap_analysis
chain. Needs a real OpenAI API key and a built index (run scripts/build_index.py
first), so it's skipped automatically when no key is configured.

    pytest tests/test_gap_analysis_smoke.py -m requires_api -v
"""
import os

import pytest

from career_copilot.graph.gap_analysis import gap_analysis
from career_copilot.graph.parse_jd import parse_jd
from career_copilot.graph.retrieve_evidence import all_chunk_ids, retrieve_evidence

pytestmark = pytest.mark.requires_api

needs_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set — skipping live API test"
)

SAMPLE_JD = """
Data Scientist — Acme Analytics

Requirements:
- 3+ years of experience with Python and SQL
- Experience deploying models to production (Docker, cloud platforms like AWS or GCP)

Nice to have:
- Experience with LangGraph or other agent orchestration frameworks
"""


@needs_key
def test_full_chain_produces_grounded_gap_report():
    jd = parse_jd(SAMPLE_JD)
    bundles = retrieve_evidence(jd)
    report = gap_analysis(jd, bundles)

    valid_ids = all_chunk_ids(bundles)
    cited = {
        cid
        for item in (report.matched + report.partial)
        for cid in item.evidence_chunk_ids
    }
    # every citation must trace back to something actually retrieved
    assert cited <= valid_ids
    # The report should preserve the parsed requirement set and return a usable summary.
    all_requirements = {i.requirement for i in report.matched + report.partial + report.missing}
    assert len(all_requirements) > 0
    assert len(report.overall_fit_summary) > 0
