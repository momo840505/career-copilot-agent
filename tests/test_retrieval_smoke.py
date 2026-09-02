"""End-to-end retrieval smoke test — this ONE test needs a real OpenAI API key
(it calls the embeddings API), so it's marked and skipped automatically when no
key is configured. Run it explicitly once you've built the index:

    pytest tests/test_retrieval_smoke.py -m requires_api -v
"""
import os

import pytest

from career_copilot.config import get_settings
from career_copilot.rag.retriever import search

pytestmark = pytest.mark.requires_api

needs_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set — skipping live API test"
)


@needs_key
def test_sql_query_retrieves_flight_or_lakehouse_project():
    settings = get_settings()
    hits = search(settings, "Does she have SQL and PostgreSQL experience?", k=4)

    assert len(hits) > 0
    hit_doc_ids = {h.doc_id for h in hits}
    # SQL/PostgreSQL is most concentrated in these two project docs
    assert hit_doc_ids & {
        "project_flight_reliability_platform",
        "project_cyber_risk_intelligence_lakehouse",
        "skills",
    }
