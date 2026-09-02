"""Loader tests — pure logic, no API key needed."""
from pathlib import Path

from career_copilot.rag.loader import load_portfolio_docs


def test_loads_all_portfolio_docs():
    portfolio_dir = Path(__file__).resolve().parents[1] / "src" / "career_copilot" / "data" / "portfolio"
    docs = load_portfolio_docs(portfolio_dir)

    # resume_profile + work_experience + skills + 5 project write-ups = 8 docs
    assert len(docs) == 8

    ids = {d.doc_id for d in docs}
    assert "resume_profile" in ids
    assert "project_retail_demand_forecasting" in ids


def test_frontmatter_parsed_into_metadata():
    portfolio_dir = Path(__file__).resolve().parents[1] / "src" / "career_copilot" / "data" / "portfolio"
    docs = load_portfolio_docs(portfolio_dir)
    project_docs = [d for d in docs if d.doc_id.startswith("project_")]

    assert len(project_docs) == 5
    for d in project_docs:
        assert "project" in d.tags
        assert d.title  # not empty
        assert len(d.content) > 100  # body actually has substance
