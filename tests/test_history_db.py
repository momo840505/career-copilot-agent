"""Pure-logic tests for career_copilot.api.db (Phase 7a) — no API key needed, no
FastAPI involved. Each test gets its own throwaway SQLite file via pytest's tmp_path
fixture, so tests never touch the real history.db and can't interfere with each other.
"""
from __future__ import annotations

from career_copilot.api.db import get_history, init_db, insert_history, list_history


def test_insert_then_get_round_trips_all_fields(tmp_path):
    db_path = tmp_path / "history.db"
    init_db(db_path)

    inserted = insert_history(
        db_path, "client-a", "gap_analysis", "Data Analyst", "some JD text", {"overall_fit_summary": "ok"}
    )

    fetched = get_history(db_path, "client-a", inserted.id)
    assert fetched is not None
    assert fetched.id == inserted.id
    assert fetched.client_id == "client-a"
    assert fetched.kind == "gap_analysis"
    assert fetched.job_title == "Data Analyst"
    assert fetched.jd_text == "some JD text"
    assert fetched.result == {"overall_fit_summary": "ok"}
    assert fetched.created_at  # non-empty ISO timestamp


def test_get_history_returns_none_for_unknown_id(tmp_path):
    db_path = tmp_path / "history.db"
    init_db(db_path)
    assert get_history(db_path, "client-a", "does-not-exist") is None


def test_get_history_returns_none_for_wrong_client_id(tmp_path):
    """A record belonging to client-a must not be readable by client-b guessing the
    id — this is the one real ownership boundary client_id provides (see db.py's
    module docstring on what it isn't: a security mechanism against a determined
    attacker, since client_id itself is just a self-reported header)."""
    db_path = tmp_path / "history.db"
    init_db(db_path)
    record = insert_history(db_path, "client-a", "draft", "Engineer", "jd", {"body": "..."})
    assert get_history(db_path, "client-b", record.id) is None


def test_list_history_scoped_to_client_id(tmp_path):
    db_path = tmp_path / "history.db"
    init_db(db_path)
    insert_history(db_path, "client-a", "gap_analysis", "A Job", "jd a", {})
    insert_history(db_path, "client-b", "gap_analysis", "B Job", "jd b", {})

    a_records = list_history(db_path, "client-a")
    assert len(a_records) == 1
    assert a_records[0].job_title == "A Job"

    b_records = list_history(db_path, "client-b")
    assert len(b_records) == 1
    assert b_records[0].job_title == "B Job"


def test_list_history_most_recent_first(tmp_path):
    db_path = tmp_path / "history.db"
    init_db(db_path)
    first = insert_history(db_path, "client-a", "gap_analysis", "First", "jd", {})
    second = insert_history(db_path, "client-a", "gap_analysis", "Second", "jd", {})

    records = list_history(db_path, "client-a")
    assert [r.id for r in records] == [second.id, first.id] or records[0].created_at >= records[1].created_at


def test_init_db_is_idempotent(tmp_path):
    """Called on every app startup (see api/app.py's lifespan) -- must not error or
    wipe existing data on a second call."""
    db_path = tmp_path / "history.db"
    init_db(db_path)
    insert_history(db_path, "client-a", "gap_analysis", "Job", "jd", {})
    init_db(db_path)  # second call, e.g. a restart
    assert len(list_history(db_path, "client-a")) == 1
