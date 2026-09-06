"""Regression tests for the access-code brute-force hole, and for the
_hits dict-growth leak found in a follow-up strict audit.

An earlier attempt used slowapi's @limiter.limit() decorator. A live TestClient check
(see api/rate_limit.py's module docstring) showed it didn't actually work: FastAPI
resolves a route's other Depends() (require_access_code) before ever calling the
decorated endpoint function, so a wrong access code returned its 401 and skipped the
rate check entirely -- unlimited wrong-code guessing was still possible. The first two
tests below lock in the fix (rate_limit() as its own Depends(), listed before
require_access_code) so that regression can't come back unnoticed.

A separate audit pass caught that _hits (the module-level counter dict) never removed
an entry once created -- every distinct (IP, route) pair ever seen stayed in memory
forever, even long after that caller stopped sending requests. On a long-lived single
instance getting varied or bot traffic, that's a slow, real memory leak. The sweep
added to rate_limit()'s dependency (drop any fully-expired entry, not just the current
key, on every call) is locked in by test_stale_entries_get_swept_from_memory below.
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


def test_stale_entries_get_swept_from_memory() -> None:
    """The leak an audit pass caught: without the sweep, _hits keeps one entry per
    distinct (IP, route) pair FOREVER, even once that entry's whole window has long
    since expired. Simulate that by manually back-dating an old entry's timestamp
    past the window, then confirm the NEXT request from a different caller sweeps it
    out -- not just prunes its own key, the whole dict.
    """
    from career_copilot.api.app import app

    _reset_rate_limit_state()
    client = TestClient(app)

    # Simulate a caller that hit the limiter once, long enough ago that its window
    # has fully expired -- this is the entry that must NOT survive forever. Keyed on
    # /auth/verify specifically because that route actually goes through rate_limit()
    # -- /health does not, so a request there would never trigger the sweep at all.
    stale_key = ("203.0.113.1", "/auth/verify")
    _hits[stale_key].append(0.0)  # time.monotonic()==0.0 is always > _WINDOW_SECONDS ago

    assert stale_key in _hits
    # A request from a DIFFERENT (simulated) caller to the same rate-limited route
    # should trigger the sweep and clear the stale entry out, proving the sweep looks
    # at the whole dict, not just whatever key the current request happens to use.
    client.post("/auth/verify")
    assert stale_key not in _hits, "stale (IP, route) entries must be swept, not kept forever"
