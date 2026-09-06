"""Phase 8: a small hand-rolled per-IP rate limiter.

Tried slowapi's `@limiter.limit()` decorator first (see git history) -- a live
TestClient check showed it doesn't actually protect what it needs to: FastAPI
resolves a route's `Depends()` parameters (`require_access_code` included) BEFORE
ever calling the decorated endpoint function, so a WRONG access code raises its 401
and returns before the decorator's own limit-check code runs at all. Confirmed with a
script: 7 wrong-code calls to /auth/verify in a row came back 401 seven times, never a
429 -- unlimited wrong-code guessing was still possible, which is exactly the attack
this exists to stop.

The fix is to make the rate limit itself a `Depends()`, not a decorator around the
whole function -- that way it's resolved by the same mechanism, in the order it's
declared, and can run (and raise) BEFORE `require_access_code` gets a chance to. Every
protected route below lists `rate_limit(...)` FIRST in its parameters for exactly that
reason.

In-memory fixed-window counter per (client IP, route path) -- no Redis, no external
service, matching every other single-instance-appropriate choice already made in this
project (the in-memory LangGraph checkpointer in build_graph.py, the in-memory
/metrics counters in observability.py). It resets on process restart, which is fine
for a single-instance personal-portfolio demo: a restart is effectively "the clock
resets" for an attacker too.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60.0
_lock = Lock()
_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """request.client.host is only the REAL caller's IP if uvicorn is told to trust
    the reverse proxy in front of it -- see docker/entrypoint.sh's --proxy-headers /
    --forwarded-allow-ips flags. Without those, every request behind Render would
    show the same proxy IP, and this limiter would end up rate-limiting all visitors
    together instead of isolating one abusive client. request.client can be None in
    some ASGI test setups, hence the fallback.
    """
    return request.client.host if request.client else "unknown"


def rate_limit(max_per_minute: int):
    """Returns a FastAPI dependency that enforces `max_per_minute` requests per
    client IP for whichever route it's attached to. Each route gets its own
    independent counter (keyed by IP + route path), so hammering /draft doesn't
    also lock the same caller out of /auth/verify.
    """

    def _dependency(request: Request) -> None:
        key = (_client_ip(request), request.url.path)
        now = time.monotonic()
        with _lock:
            # Opportunistic sweep: drop any (IP, route) entry whose whole window has
            # already expired -- not just this request's own key. Without this,
            # _hits gains one entry per distinct (IP, route) pair ever seen and NEVER
            # shrinks again, even long after that IP stops sending requests -- a real,
            # slow memory leak on a long-lived single instance getting varied or bot
            # traffic (Render's free tier spinning down after 15 idle minutes hides
            # this in casual testing, but it's still a real bug). A single-instance
            # in-memory limiter doesn't need anything fancier than a sweep like this.
            stale_keys = [
                k for k, v in _hits.items() if not v or now - v[-1] > _WINDOW_SECONDS
            ]
            for k in stale_keys:
                del _hits[k]

            hits = _hits[key]
            while hits and now - hits[0] > _WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= max_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_per_minute} requests per minute per IP.",
                )
            hits.append(now)

    return _dependency
