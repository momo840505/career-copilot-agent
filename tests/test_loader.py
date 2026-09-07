from pathlib import Path

from career_copilot.rag.loader import load_portfolio_docs


def _portfolio_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "career_copilot"
        / "data"
        / "portfolio"
    )


def test_loads_all_portfolio_docs():
    docs = load_portfolio_docs(_portfolio_dir())

    assert len(docs) == 9
    ids = {doc.doc_id for doc in docs}
    assert "resume_profile" in ids
    assert "project_retail_demand_forecasting" in ids
    assert "project_career_copilot_agent" in ids


def test_frontmatter_parsed_into_metadata():
    docs = load_portfolio_docs(_portfolio_dir())
    project_docs = [doc for doc in docs if doc.doc_id.startswith("project_")]

    assert len(project_docs) == 6
    for doc in project_docs:
        assert "project" in doc.tags
        assert doc.title
        assert len(doc.content) > 100
