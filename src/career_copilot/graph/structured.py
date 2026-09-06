"""Shared helper: call an LLM constrained to a Pydantic schema, and if it produces
something that still fails our validators, feed the error back and let the model
repair its own output instead of just crashing.

parse_jd, gap_analysis, draft_writer, and critic all go through this one helper,
so the retry-with-feedback logic only lives in one place.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from career_copilot.observability import metrics

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the model still hasn't produced valid output after every repair
    attempt is used up."""


# Only shown when an attempt's error text is identical to the previous attempt's —
# see the "stuck" logic in invoke_structured below.
_STUCK_RETRY_NOTE = (
    "\n\nIMPORTANT: this is the exact same error as your previous attempt — repeating the "
    "same fix will just produce the same wrong answer again. Do not merely reword your "
    "output. Make a genuinely different, definitive decision this time: if you're torn "
    "between two options, commit to exactly one; if you cannot support something at all, "
    "remove it entirely rather than trying to word your way around the problem."
)


def _with_bumped_temperature(llm: BaseChatModel) -> BaseChatModel:
    """Return a copy of `llm` at a higher temperature, for the one retry right after
    we've detected it's stuck repeating itself (see invoke_structured below).

    Uses `model_copy(update=...)` rather than `llm.bind(temperature=...)` on purpose —
    `.bind()` chains through a RunnableBinding wrapper that has known bugs with
    `with_structured_output` silently dropping the override (langchain-ai/langchain
    #23167, #35320). model_copy just swaps the field on a fresh instance instead.

    If some provider doesn't support model_copy the way we expect, this just falls
    back to the original llm — the escalated wording in _STUCK_RETRY_NOTE still helps
    on its own even without the temperature bump.
    """
    try:
        return llm.model_copy(update={"temperature": 0.7})  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - provider-specific, fall back either way
        return llm


def _log_success(node_name: str, attempt: int, stuck_ever: bool, started_at: float) -> None:
    """One INFO log line plus a metrics update for a successful call, shared by both
    success returns below so GET /metrics sees both a clean first try (attempt == 1)
    and a success-after-repair without duplicating this at every return site."""
    duration_ms = (time.monotonic() - started_at) * 1000
    metrics.record_llm_call(
        node_name,
        attempts=attempt,
        stuck_occurred=stuck_ever,
        succeeded=True,
        duration_ms=duration_ms,
    )
    logger.info(
        "invoke_structured: node=%s succeeded on attempt %d (%.0fms)%s",
        node_name,
        attempt,
        duration_ms,
        " after a STUCK retry" if stuck_ever else "",
        extra={"node": node_name, "attempt": attempt, "duration_ms": round(duration_ms, 1), "stuck_occurred": stuck_ever},
    )


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    max_retries: int = 2,
    validate: Callable[[T], T] | None = None,
    node_name: str = "unknown",
) -> T:
    """Invoke `llm`, constrained to `schema`, with up to `max_retries` repair attempts.

    Catches Exception broadly rather than just pydantic.ValidationError because the
    exact error LangChain raises for a schema-invalid response depends on the version
    and on which with_structured_output method is active. Whatever the error, it gets
    appended as a follow-up message asking the model to fix exactly that, then we
    try again.

    `validate`, if given, runs after the schema's own field/model validators, which
    only ever see the one object being built with no outside context. Use it for
    checks that need something the schema can't see by itself — e.g. gap_analysis
    uses it to check that every chunk_id a report cites actually exists in the
    retrieved evidence, since only the caller knows what a valid chunk_id looks like.
    A raise inside `validate` is treated the same as a schema failure and feeds back
    into the same retry loop.

    Stuck retries: most nodes run at temperature=0.0 for reproducibility, but that
    means if the repair message is itself deterministic (same error -> same wording),
    the model can regenerate the exact same wrong answer every attempt and burn the
    whole retry budget without changing anything (this actually happened on the
    golden eval set — 3/3 identical failures, twice, on two different nodes). So each
    attempt's error text gets compared to the previous one's; on an exact repeat, the
    next attempt gets an explicit "you're stuck, make a different decision" note
    (_STUCK_RETRY_NOTE) and runs at a bumped temperature to actually break the
    determinism instead of just asking nicely.

    `node_name` only affects observability (career_copilot/observability.py) — it
    labels this call in the logs and in GET /metrics' per-node breakdown, so you can
    tell which node is burning retries. Omitting it just buckets the call under
    "unknown"; behavior is otherwise identical.
    """
    structured_llm = llm.with_structured_output(schema)
    base_messages = list(messages)
    attempt_messages = base_messages
    last_error: Exception | None = None
    last_error_text: str | None = None
    stuck = False
    stuck_ever = False
    started_at = time.monotonic()

    for attempt in range(1, max_retries + 2):  # 1 initial try + max_retries repairs
        current_structured_llm = (
            _with_bumped_temperature(llm).with_structured_output(schema) if stuck else structured_llm
        )
        try:
            result = current_structured_llm.invoke(attempt_messages)
        except Exception as e:  # noqa: BLE001 - error shape varies, see docstring
            # Raw output didn't even parse against the schema, so there's nothing
            # concrete to show back other than the error itself.
            last_error = e
            error_text = str(e)
            stuck = last_error_text is not None and error_text == last_error_text
            stuck_ever = stuck_ever or stuck
            last_error_text = error_text
            logger.warning(
                "invoke_structured: attempt %d/%d failed schema parsing%s: %s",
                attempt,
                max_retries + 1,
                " (STUCK — repeats previous error)" if stuck else "",
                e,
                extra={"node": node_name, "attempt": attempt, "stuck": stuck, "failure_kind": "schema"},
            )
            attempt_messages = base_messages + [
                HumanMessage(
                    content=(
                        "Your previous response did not satisfy the required schema. "
                        f"Validation error:\n{e}\n\n"
                        "Return a corrected response that fixes exactly this error. "
                        "Do not change anything else."
                        + (_STUCK_RETRY_NOTE if stuck else "")
                    )
                )
            ]
            continue

        if validate is None:
            _log_success(node_name, attempt, stuck_ever, started_at)
            return result  # type: ignore[return-value]

        try:
            validated = validate(result)
            _log_success(node_name, attempt, stuck_ever, started_at)
            return validated
        except Exception as e:  # noqa: BLE001 - error shape varies, see docstring
            last_error = e
            error_text = str(e)
            stuck = last_error_text is not None and error_text == last_error_text
            stuck_ever = stuck_ever or stuck
            last_error_text = error_text
            logger.warning(
                "invoke_structured: attempt %d/%d failed extra validation%s: %s",
                attempt,
                max_retries + 1,
                " (STUCK — repeats previous error)" if stuck else "",
                e,
                extra={"node": node_name, "attempt": attempt, "stuck": stuck, "failure_kind": "validate"},
            )
            # Show the model its own previous (schema-valid but rule-violating) output
            # alongside the error, so the next attempt is a targeted edit instead of a
            # blind re-roll. Without this it has no memory of what it actually wrote
            # and tends to repeat the same mistake, or trade it for a different one —
            # we saw it keep shuffling one requirement between matched/partial/missing
            # on every attempt.
            attempt_messages = base_messages + [
                AIMessage(content=result.model_dump_json()),
                HumanMessage(
                    content=(
                        "The response above satisfies the schema but violates an "
                        f"additional rule. Error:\n{e}\n\n"
                        "Return a corrected version of YOUR PREVIOUS RESPONSE above that "
                        "fixes exactly this problem. Keep everything else the same."
                        + (_STUCK_RETRY_NOTE if stuck else "")
                    )
                ),
            ]

    duration_ms = (time.monotonic() - started_at) * 1000
    metrics.record_llm_call(
        node_name,
        attempts=max_retries + 1,
        stuck_occurred=stuck_ever,
        succeeded=False,
        duration_ms=duration_ms,
    )
    logger.error(
        "invoke_structured: node=%s exhausted %d attempts, giving up",
        node_name,
        max_retries + 1,
        extra={"node": node_name, "attempts": max_retries + 1, "stuck_occurred": stuck_ever},
    )
    raise StructuredOutputError(
        f"Failed to get valid {schema.__name__} output after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
