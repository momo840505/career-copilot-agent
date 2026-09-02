"""Shared helper: call an LLM constrained to a Pydantic schema, and if it produces
something that still fails our validators, feed the error back and let the model
repair its own output — instead of silently accepting bad data or crashing the run.

Every node from Phase 2 onward (parse_jd, gap_analysis, draft_writer, critic) uses this
same helper, so the retry-with-feedback behavior only has to be built and tested once.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the model still fails to produce schema-valid output after every
    repair attempt has been exhausted."""


# Shown only when an attempt's error is BYTE-IDENTICAL to the immediately preceding
# attempt's error — see the "stuck retry" note in invoke_structured's docstring.
_STUCK_RETRY_NOTE = (
    "\n\nIMPORTANT: this is the exact same error as your previous attempt — repeating the "
    "same fix will just produce the same wrong answer again. Do not merely reword your "
    "output. Make a genuinely different, definitive decision this time: if you're torn "
    "between two options, commit to exactly one; if you cannot support something at all, "
    "remove it entirely rather than trying to word your way around the problem."
)


def _with_bumped_temperature(llm: BaseChatModel) -> BaseChatModel:
    """Best-effort: return a copy of `llm` at a higher temperature, for the one retry
    attempt right after we've detected the model is stuck repeating itself (see
    invoke_structured's docstring). LangChain chat models are themselves Pydantic
    models, so `model_copy(update=...)` swaps the field directly on a fresh instance —
    deliberately NOT `llm.bind(temperature=...)`, which chains through a RunnableBinding
    wrapper that has had real, documented interaction bugs with `with_structured_output`
    (e.g. langchain-ai/langchain#23167, #35320) silently dropping the override. If this
    ever fails for some provider that doesn't support model_copy the way we expect, fall
    back to the original llm rather than raising — the escalated wording in
    _STUCK_RETRY_NOTE still helps even without the temperature bump.
    """
    try:
        return llm.model_copy(update={"temperature": 0.7})  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - deliberately broad, see docstring
        return llm


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    max_retries: int = 2,
    validate: Callable[[T], T] | None = None,
) -> T:
    """Invoke `llm`, constrained to `schema`, with up to `max_retries` repair attempts.

    We catch broadly (not just pydantic.ValidationError) because the exact exception
    LangChain raises when the model's JSON fails schema validation varies by version
    and by which `with_structured_output` method is in use. Any failure on attempt N
    gets its error message appended as a follow-up turn asking the model to fix
    exactly that problem, then we try again.

    `validate`, if given, runs AFTER the schema's own field/model validators — which
    only ever see the one object being built, with no outside context. Use `validate`
    for checks that need something the schema can't see by itself, e.g. "does every
    chunk_id this report cites actually exist in the evidence we retrieved?" (a real
    example from gap_analysis — the schema has no way to know what a valid chunk_id
    looks like, only the caller does). Return the result on success, or raise on
    failure; a raise is treated exactly like a schema validation failure and feeds
    back into the same retry loop.

    STUCK RETRIES: most nodes run at temperature=0.0 (see llm.py), which is normally
    what you want — reproducible output. But it has a real failure mode here: if the
    repair message we send back is itself deterministic (same error -> same wording),
    a temperature-0 model can regenerate the EXACT SAME wrong answer on every attempt,
    burning the whole retry budget without ever trying anything different (observed
    live on the golden eval set — 3/3 identical failures, twice, on two different
    nodes). We detect this by comparing each attempt's error text to the previous
    attempt's: on a byte-identical repeat, the next attempt gets an explicit "you're
    stuck, make a different decision" note (_STUCK_RETRY_NOTE) AND runs at a bumped
    temperature (_with_bumped_temperature) to actually break the determinism, not just
    ask nicely.
    """
    structured_llm = llm.with_structured_output(schema)
    base_messages = list(messages)
    attempt_messages = base_messages
    last_error: Exception | None = None
    last_error_text: str | None = None
    stuck = False

    for attempt in range(1, max_retries + 2):  # 1 initial try + max_retries repairs
        current_structured_llm = (
            _with_bumped_temperature(llm).with_structured_output(schema) if stuck else structured_llm
        )
        try:
            result = current_structured_llm.invoke(attempt_messages)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            # The model's raw output didn't even parse against the schema, so we have
            # nothing concrete to show it back — fall back to a plain error message.
            last_error = e
            error_text = str(e)
            stuck = last_error_text is not None and error_text == last_error_text
            last_error_text = error_text
            logger.warning(
                "invoke_structured: attempt %d/%d failed schema parsing%s: %s",
                attempt,
                max_retries + 1,
                " (STUCK — repeats previous error)" if stuck else "",
                e,
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
            return result  # type: ignore[return-value]

        try:
            return validate(result)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = e
            error_text = str(e)
            stuck = last_error_text is not None and error_text == last_error_text
            last_error_text = error_text
            logger.warning(
                "invoke_structured: attempt %d/%d failed extra validation%s: %s",
                attempt,
                max_retries + 1,
                " (STUCK — repeats previous error)" if stuck else "",
                e,
            )
            # Crucial: show the model its OWN previous (schema-valid but rule-violating)
            # output before the error, so the next attempt is a targeted edit — not a
            # blind re-roll from the original prompt with an abstract hint. Without this,
            # the model has no memory of what it actually wrote and tends to repeat, or
            # trade, the same mistake across retries (observed live: it kept
            # re-duplicating a *different* requirement across matched/partial/missing
            # on every attempt).
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

    raise StructuredOutputError(
        f"Failed to get valid {schema.__name__} output after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
