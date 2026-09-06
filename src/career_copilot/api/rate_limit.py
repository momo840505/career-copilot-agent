"""A small hand-rolled per-IP rate limiter.

I tried slowapi's `@limiter.limit()` decorator first, but a live TestClient check
showed it doesn't actually protect anything: FastAPI resolves a route's Depends()
(require_access_code included) before ever calling the decorated endpoint function,
so a wrong access code raises its 401 and returns before the decorator's own
limit-check code runs at all. Confirmed with a script -- 7 wrong-code calls to
/auth/verify in a row came back 401 seven times, never a 429. Unlimited wrong-code
guessing was still possible, which is exactly the attack this is supposed to stop.

The fix is to make the rate limit itself a Depends(), not a decorator around the
whole function -- that way it resolves through the same mechanism, in the order it's
declared, and can run (and raise) before require_access_code gets a chance to. Every
protected route lists rate_limit(...) first in its parameters for that reason.

In-memory fixed-window counter per (client IP, route path), no Redis or external
service, same single-instance-appropriate choice as the in-memory LangGraph
checkpointer in build_graph.py and the in-memory /metrics counters in
observability.py. It resets on process restart, which is fine for a single-instance
portfolio demo -- a restart resets the clock for an attacker too.
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
            # already expired, not just this request's own key. Without this, _hits
            # gains one entry per distinct (IP, route) pair ever seen and never
            # shrinks again, even after that IP stops sending requests -- a slow
            # memory leak on a long-lived instance getting varied or bot traffic.
            # Render's free tier spinning down after 15 idle minutes hides this in
            # casual testing, but it's still a real bug. A sweep like this is enough
            # for a single-instance in-memory limiter.
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
