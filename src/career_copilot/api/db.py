"""Phase 7a: SQLite persistence for the site's "history" feature (past gap-analysis
and cover-letter runs).

Deliberately plain stdlib `sqlite3`, no ORM: one small table, a handful of queries,
and this project already leans toward the smallest tool that does the job (see
`rag/store.py`'s own comment about using chromadb's client directly). A fresh
connection is opened per call rather than shared/pooled — SQLite connections aren't
thread-safe by default, and FastAPI runs sync route handlers in a thread pool, so
sharing one connection across requests would need explicit locking for no real
benefit at this app's scale. SQLite's own file-level locking handles the rest.

There's no per-user account system (see api/auth.py for why: one shared access code,
not real login) — `client_id` is a random id the frontend generates once and keeps in
localStorage, purely to give each browser its own "my history" view. It is NOT a
security boundary: anyone who can call the API at all can pass any client_id they
like. Treat it as a UI convenience, not an auth mechanism.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Safe to call every time the app starts — CREATE TABLE IF NOT EXISTS."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                job_title TEXT NOT NULL,
                jd_text TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_client_id ON history(client_id)")


@dataclass
class HistoryRecord:
    id: str
    client_id: str
    kind: str  # "gap_analysis" | "draft"
    job_title: str
    jd_text: str
    result: dict[str, Any]
    created_at: str  # ISO 8601 UTC


def insert_history(
    db_path: Path,
    client_id: str,
    kind: str,
    job_title: str,
    jd_text: str,
    result: dict[str, Any],
) -> HistoryRecord:
    record = HistoryRecord(
        id=str(uuid.uuid4()),
        client_id=client_id,
        kind=kind,
        job_title=job_title,
        jd_text=jd_text,
        result=result,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO history (id, client_id, kind, job_title, jd_text, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.client_id,
                record.kind,
                record.job_title,
                record.jd_text,
                json.dumps(result),
                record.created_at,
            ),
        )
    return record


def list_history(db_path: Path, client_id: str) -> list[HistoryRecord]:
    """Most recent first. Scoped to one client_id — never returns another
    browser's history (see the module docstring on what client_id is/isn't)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, client_id, kind, job_title, jd_text, result_json, created_at "
            "FROM history WHERE client_id = ? ORDER BY created_at DESC",
            (client_id,),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def get_history(db_path: Path, client_id: str, record_id: str) -> HistoryRecord | None:
    """None if the id doesn't exist OR belongs to a different client_id — a caller
    can't tell the two cases apart, which is the point (no existence leak)."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, client_id, kind, job_title, jd_text, result_json, created_at "
            "FROM history WHERE id = ? AND client_id = ?",
            (record_id, client_id),
        ).fetchone()
    return _row_to_record(row) if row else None


def _row_to_record(row: sqlite3.Row) -> HistoryRecord:
    return HistoryRecord(
        id=row["id"],
        client_id=row["client_id"],
        kind=row["kind"],
        job_title=row["job_title"],
        jd_text=row["jd_text"],
        result=json.loads(row["result_json"]),
        created_at=row["created_at"],
    )
