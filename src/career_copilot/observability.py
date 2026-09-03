"""Phase 7e: structured logging + a lightweight in-process metrics registry.

Two things live here:

1. `configure_logging()` -- sets up the root logger once, at process startup, with
   either a JSON formatter (LOG_FORMAT=json, the default in the Docker image -- see
   docker/entrypoint.sh) or a plain human-readable formatter (the default in local
   dev, where a real terminal is more pleasant to read than JSON lines). Render's
   dashboard just tails stdout, so JSON-per-line is the right shape there: it's
   greppable, and if this ever outgrows Render's own Logs tab it's already shaped to
   pipe straight into a real log aggregator without changing any application code.

2. `metrics` -- a process-wide, thread-safe counter registry. Render's free tier runs
   exactly one instance with no persistent disk, so anything fancier than in-memory
   counters (Prometheus, a real time-series DB) would be solving a problem this
   deployment doesn't have. What it tracks answers the two questions an operator
   actually asks first when something looks wrong: "is the API taking traffic and
   returning errors?" (per-route request count / latency / status-code breakdown) and
   "is the LLM pipeline healthy, or is it burning retries?" (per-node call / retry /
   STUCK / failure counts -- the STUCK counter in particular is exactly the failure
   mode `graph/structured.py`'s invoke_structured docstring describes: a temperature-0
   model regenerating the identical wrong answer instead of actually retrying).
   GET /metrics (api/app.py) serializes a snapshot of this as JSON.

Kept intentionally small: no persistence, no percentiles, no external dependency.
Restarting the process resets it -- same as Render restarting the container resets
everything else non-persistent (the Chroma index, history.db) unless a paid disk is
attached.
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
    whatever structured fields the caller passed via `extra={...}` (route, status_code,
    duration_ms, node, attempts, ...). Anything already on a stock LogRecord (the
    reserved keys below) is skipped so it isn't duplicated under its raw attribute
    name."""

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
    """Idempotent -- safe to call from api/app.py at import time AND from any CLI
    script's entry point without risking duplicate handlers (e.g. under uvicorn's
    --reload, which re-imports the app module in a fresh subprocess anyway, but a
    stray second call in the same process must not double every log line).

    LOG_LEVEL defaults to INFO. LOG_FORMAT defaults to "plain" (readable in a local
    terminal); the Docker image sets LOG_FORMAT=json (see docker/entrypoint.sh) so
    Render's log tail is one grep-able/parseable JSON object per line.
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
    """Thread-safe: uvicorn can (and by default does, for sync def routes like the
    ones in api/app.py) run request handlers on a thread pool even within a single
    process, so this locks on every update. Cheap enough not to matter at this
    traffic scale -- a portfolio demo, not a high-throughput service."""

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


# Module-level singleton, same pattern as get_settings() being the one place config
# is read from -- every node and the API layer import THIS instance rather than
# constructing their own, so counts actually accumulate across a request.
metrics = Metrics()
