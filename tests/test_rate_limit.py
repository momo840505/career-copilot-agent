"""Phase 8: regression test for the access-code brute-force hole.

An earlier attempt used slowapi's @limiter.limit() decorator. A live TestClient check
(see api/rate_limit.py's module docstring) showed it didn't actually work: FastAPI
resolves a route's other Depends() (require_access_code) before ever calling the
decorated endpoint function, so a wrong access code returned its 401 and skipped the
rate check entirely -- unlimited wrong-code guessing was still possible. This test
locks in the fix (rate_limit() as its own Depends(), listed before require_access_code)
so that regression can't come back unnoticed.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from career_copilot.api.rate_limit import _hits


def _reset_rate_limit_state() -> None:
    _hits.clear()


def test_wrong_access_code_still_gets_rate_limited(monkeypatch) -> None:
    """The actual bug this file exists to catch: a WRONG access code must count
    toward the limit, not bypass it."""
    monkeypatch.setenv("ACCESS_CODE", "secret123")
    _reset_rate_limit_state()
    from career_copilot.api.app import app  # imported after env var is set

    client = TestClient(app)
    statuses = [
        client.post("/auth/verify", headers={"X-Access-Code": "WRONG"}).status_code
        for _ in range(7)
    ]
    assert statuses[:5] == [401] * 5, "first 5 wrong-code attempts should just be 401"
    assert statuses[5:] == [429, 429], "6th+ attempt within the window must be 429"


def test_correct_access_code_also_gets_rate_limited(monkeypatch) -> None:
    """A leaked/shared code shouldn't be able to bypass the limit either -- the cap
    is per IP regardless of whether the code is right."""
    monkeypatch.setenv("ACCESS_CODE", "secret123")
    _reset_rate_limit_state()
    from career_copilot.api.app import app

    client = TestClient(app)
    statuses = [
        client.post("/auth/verify", headers={"X-Access-Code": "secret123"}).status_code
        for _ in range(7)
    ]
    assert statuses[:5] == [200] * 5
    assert statuses[5:] == [429, 429]


def test_rate_limit_is_per_route_not_global() -> None:
    """Hammering /auth/verify shouldn't also lock the same caller out of /health --
    each route gets its own counter (see rate_limit()'s key: IP + route path)."""
    _reset_rate_limit_state()
    from career_copilot.api.app import app

    client = TestClient(app)
    for _ in range(10):
        client.post("/auth/verify")
    assert client.get("/health").status_code == 200
