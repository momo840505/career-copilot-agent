from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import APIError
from pydantic import BaseModel

from career_copilot.observability import metrics

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_STUCK_RETRY_NOTE = (
    "\nThe same validation error occurred twice. Make a different correction "
    "instead of repeating the previous answer."
)


class StructuredOutputError(RuntimeError):
    pass


def _with_bumped_temperature(llm: BaseChatModel) -> BaseChatModel:
    try:
        return llm.model_copy(update={"temperature": 0.7})  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return llm


def _log_success(
    node_name: str,
    attempt: int,
    stuck_ever: bool,
    started_at: float,
) -> None:
    duration_ms = (time.monotonic() - started_at) * 1000
    metrics.record_llm_call(
        node_name,
        attempts=attempt,
        stuck_occurred=stuck_ever,
        succeeded=True,
        duration_ms=duration_ms,
    )
    logger.info(
        "structured output succeeded",
        extra={
            "node": node_name,
            "attempt": attempt,
            "duration_ms": round(duration_ms, 1),
            "stuck_occurred": stuck_ever,
        },
    )


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    max_retries: int = 2,
    validate: Callable[[T], T] | None = None,
    node_name: str = "unknown",
) -> T:
    structured_llm = llm.with_structured_output(schema)
    base_messages = list(messages)
    attempt_messages = base_messages
    last_error: Exception | None = None
    last_error_text: str | None = None
    stuck = False
    stuck_ever = False
    started_at = time.monotonic()

    for attempt in range(1, max_retries + 2):
        current = (
            _with_bumped_temperature(llm).with_structured_output(schema)
            if stuck
            else structured_llm
        )
        try:
            result = current.invoke(attempt_messages)
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            error_text = str(exc)
            stuck = last_error_text is not None and error_text == last_error_text
            stuck_ever = stuck_ever or stuck
            last_error_text = error_text

            logger.warning(
                "structured output parse failed",
                extra={
                    "node": node_name,
                    "attempt": attempt,
                    "stuck": stuck,
                    "failure_kind": "schema",
                },
            )
            attempt_messages = base_messages + [
                HumanMessage(
                    content=(
                        "The previous response did not satisfy the required schema.\n"
                        f"Validation error:\n{exc}\n\n"
                        "Return a corrected response."
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            error_text = str(exc)
            stuck = last_error_text is not None and error_text == last_error_text
            stuck_ever = stuck_ever or stuck
            last_error_text = error_text

            logger.warning(
                "structured output validation failed",
                extra={
                    "node": node_name,
                    "attempt": attempt,
                    "stuck": stuck,
                    "failure_kind": "validate",
                },
            )
            attempt_messages = base_messages + [
                AIMessage(content=result.model_dump_json()),
                HumanMessage(
                    content=(
                        "The response above is valid JSON but fails an application rule.\n"
                        f"Validation error:\n{exc}\n\n"
                        "Correct the previous response and keep unrelated fields unchanged."
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
        "structured output exhausted retries",
        extra={
            "node": node_name,
            "attempts": max_retries + 1,
            "stuck_occurred": stuck_ever,
        },
    )
    raise StructuredOutputError(
        f"Failed to get valid {schema.__name__} output after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
