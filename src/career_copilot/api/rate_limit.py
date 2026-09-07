from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60.0
_lock = Lock()
_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit(max_per_minute: int):
    def _dependency(request: Request) -> None:
        key = (_client_ip(request), request.url.path)
        now = time.monotonic()

        with _lock:
            stale = [
                item
                for item, values in _hits.items()
                if not values or now - values[-1] > _WINDOW_SECONDS
            ]
            for item in stale:
                del _hits[item]

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
