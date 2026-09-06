"""Structured logging plus a small in-process metrics registry.

configure_logging() sets up the root logger once at process startup -- either a JSON
formatter (LOG_FORMAT=json, what the Docker image uses, see docker/entrypoint.sh) or a
plain human-readable one (the local-dev default, since a real terminal is nicer to
read than JSON lines). Render just tails stdout, so JSON-per-line is greppable there
and would also pipe straight into a real log aggregator later without touching any
application code.

`metrics` is a process-wide, thread-safe counter registry. Render's free tier runs one
instance with no persistent disk, so Prometheus or a real time-series DB would be
solving a problem this doesn't have -- in-memory counters are enough. It tracks the
two things worth checking when something looks wrong: is the API taking traffic and
returning errors (per-route count / latency / status-code breakdown), and is the LLM
pipeline healthy or burning retries (per-node call / retry / STUCK / failure counts --
STUCK here is the same stuck-retry case graph/structured.py's invoke_structured deals
with). GET /metrics in api/app.py just serializes a snapshot of this as JSON.

Kept small on purpose: no persistence, no percentiles, no external dependency.
Restarting the process resets it, same as restarting the container resets everything
else non-persistent (the Chroma index, history.db) unless a paid disk is attached.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field

# --- 1. logging setup -------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """One JSON object per log line: timestamp, level, logger name, message, plus
    whatever extra fields the caller passed (route, status_code, duration_ms, node,
    attempts, ...). Skips anything already on a stock LogRecord so it doesn't get
    duplicated under its raw attribute name."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure_logging() -> None:
    """Idempotent -- safe to call from both api/app.py at import time and any CLI
    script's entry point without doubling up handlers if it somehow gets called twice
    in the same process.

    LOG_LEVEL defaults to INFO. LOG_FORMAT defaults to "plain" (readable locally); the
    Docker image sets LOG_FORMAT=json (see docker/entrypoint.sh) so Render's log tail
    is one parseable JSON object per line.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "plain").lower()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]


# --- 2. metrics --------------------------------------------------------------------


@dataclass
class _NodeStats:
    calls: int = 0
    retries: int = 0
    stuck_retries: int = 0
    failures: int = 0
    total_duration_ms: float = 0.0

    def as_dict(self) -> dict:
        avg = self.total_duration_ms / self.calls if self.calls else 0.0
        return {
            "calls": self.calls,
            "retries": self.retries,
            "stuck_retries": self.stuck_retries,
            "failures": self.failures,
            "avg_duration_ms": round(avg, 1),
        }


@dataclass
class _RouteStats:
    count: int = 0
    total_duration_ms: float = 0.0
    status_counts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        avg = self.total_duration_ms / self.count if self.count else 0.0
        return {
            "count": self.count,
            "avg_duration_ms": round(avg, 1),
            "status_counts": dict(self.status_counts),
        }


class Metrics:
    """Thread-safe: uvicorn runs sync def routes (like the ones in api/app.py) on a
    thread pool even within one process, so this locks on every update. That's cheap
    enough not to matter at this traffic scale -- a portfolio demo, not a
    high-throughput service."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._routes: dict[str, _RouteStats] = {}
        self._nodes: dict[str, _NodeStats] = {}

    def record_request(self, route: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            stats = self._routes.setdefault(route, _RouteStats())
            stats.count += 1
            stats.total_duration_ms += duration_ms
            key = str(status_code)
            stats.status_counts[key] = stats.status_counts.get(key, 0) + 1

    def record_llm_call(
        self,
        node: str,
        *,
        attempts: int,
        stuck_occurred: bool,
        succeeded: bool,
        duration_ms: float,
    ) -> None:
        with self._lock:
            stats = self._nodes.setdefault(node, _NodeStats())
            stats.calls += 1
            stats.retries += max(attempts - 1, 0)
            if stuck_occurred:
                stats.stuck_retries += 1
            if not succeeded:
                stats.failures += 1
            stats.total_duration_ms += duration_ms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "routes": {k: v.as_dict() for k, v in self._routes.items()},
                "llm_nodes": {k: v.as_dict() for k, v in self._nodes.items()},
            }


# Module-level singleton, same idea as get_settings() -- every node and the API layer
# import this one instance instead of building their own, so counts actually add up
# across requests.
metrics = Metrics()
