"""A shared access-code gate, not real per-user login.

Deliberately the simplest thing that actually addresses the real risk: once this
service is deployed with a public URL, every request runs against MY real OpenAI API
key (see llm.py / rag/store.py) -- an unprotected endpoint plus a shared link is
enough for a stranger to run up a real bill. A single shared secret checked on every
request closes that hole without needing accounts, password hashing, or session
management, which would be real added complexity for a single-person portfolio demo
with no actual multi-user data separation to protect (see api/db.py's docstring on
what client_id is/isn't).

`require_access_code` takes `settings` via FastAPI's own Depends(get_settings) rather
than calling get_settings() directly, specifically so tests can override it with
`app.dependency_overrides[get_settings] = ...` instead of monkeypatching module
globals -- the standard FastAPI testing pattern for this exact situation.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from career_copilot.config import Settings, get_settings


def require_access_code(
    x_access_code: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.access_code:
        # No ACCESS_CODE configured at all -- gate is off, matching this project's
        # existing local-dev-friendly defaults elsewhere (see llm.py raising only
        # when a key is actually needed, not at import time).
        return
    if x_access_code != settings.access_code:
        raise HTTPException(status_code=401, detail="Missing or incorrect access code.")


def get_client_id(x_client_id: str | None = Header(default=None)) -> str:
    if not x_client_id:
        raise HTTPException(status_code=400, detail="X-Client-Id header is required.")
    return x_client_id
