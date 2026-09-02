"""invoke_structured's retry-with-repair loop, exercised with fake/scripted LLMs — no
API key needed. This is what actually caught (and now locks in the fix for) the "stuck
retry" bug found by the Phase 5 golden eval set: a temperature=0 node whose repair
message is itself deterministic can regenerate the identical wrong answer on every
attempt, burning the whole retry budget on repeats instead of genuinely different tries.
"""
from __future__ import annotations

import pytest

from career_copilot.graph.structured import StructuredOutputError, invoke_structured


class _FakeResult:
    """Stands in for a Pydantic model instance — invoke_structured only ever calls
    .model_dump_json() on it, never anything schema-specific."""

    def model_dump_json(self) -> str:
        return '{"fake": true}'


class _FakeStructuredLLM:
    def __init__(self, temperature: float, invoke_fn) -> None:
        self.temperature = temperature
        self._invoke_fn = invoke_fn

    def invoke(self, messages):
        return self._invoke_fn(messages)


class _FakeLLM:
    """Duck-types just enough of BaseChatModel for invoke_structured: with_structured_output
    and model_copy. temperature is tracked so tests can assert on it directly."""

    def __init__(self, temperature: float = 0.0, invoke_fn=None) -> None:
        self.temperature = temperature
        self._invoke_fn = invoke_fn or (lambda messages: _FakeResult())

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self.temperature, self._invoke_fn)

    def model_copy(self, update=None):
        # type(self)(...), not _FakeLLM(...): real pydantic model_copy() preserves the
        # concrete subclass of the instance it's copying, and the tests below rely on
        # that (TrackingLLM subclasses _FakeLLM to observe with_structured_output calls
        # made on the copy _with_bumped_temperature produces after an escalation).
        new_temp = (update or {}).get("temperature", self.temperature)
        return type(self)(new_temp, self._invoke_fn)


def test_succeeds_immediately_when_validate_passes():
    llm = _FakeLLM()
    result = invoke_structured(llm, object, [], validate=lambda r: r)
    assert isinstance(result, _FakeResult)


def test_recovers_from_a_single_transient_validation_failure():
    calls = {"n": 0}

    def validate(result):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("minor, one-off issue")
        return result

    llm = _FakeLLM()
    result = invoke_structured(llm, object, [], max_retries=2, validate=validate)
    assert isinstance(result, _FakeResult)
    assert calls["n"] == 2


def test_raises_structured_output_error_after_exhausting_retries():
    def always_fails(result):
        raise ValueError("never satisfied")

    llm = _FakeLLM()
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, object, [], max_retries=2, validate=always_fails)


def test_stuck_on_identical_repeated_error_escalates_temperature_on_the_next_attempt():
    # The exact bug the golden eval set caught: the SAME validation error twice in a
    # row (a model stuck in a temperature=0 local minimum) should trigger a temperature
    # bump on the attempt right after the repeat is detected. invoke_structured only
    # calls with_structured_output twice in total here: once up front (reused for every
    # non-escalated attempt — it doesn't rebuild it every loop iteration), and once more
    # only for the attempt that runs right after a stuck repeat is detected.
    temps_used = []

    class TrackingLLM(_FakeLLM):
        def with_structured_output(self, schema):
            temps_used.append(self.temperature)
            return super().with_structured_output(schema)

    def always_same_error(result):
        raise ValueError("duplicate classification: X appears twice")

    llm = TrackingLLM(temperature=0.0)
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, object, [], max_retries=2, validate=always_same_error)

    # 3 total attempts (1 initial + 2 repairs). Attempts 1 and 2 fail with the IDENTICAL
    # error, so attempt 3 (the last one) is the one escalated attempt: the initial
    # (pre-loop) build at 0.0, then one more build at the bumped 0.7 for attempt 3.
    assert temps_used == [0.0, 0.7]


def test_no_escalation_when_consecutive_errors_differ():
    temps_used = []

    class TrackingLLM(_FakeLLM):
        def with_structured_output(self, schema):
            temps_used.append(self.temperature)
            return super().with_structured_output(schema)

    calls = {"n": 0}

    def different_error_each_time(result):
        calls["n"] += 1
        raise ValueError(f"a different problem #{calls['n']}")

    llm = TrackingLLM(temperature=0.0)
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, object, [], max_retries=2, validate=different_error_each_time)

    # Errors never repeat back-to-back, so escalation never triggers, so
    # with_structured_output is only ever called once — the initial pre-loop build,
    # reused unchanged for every attempt.
    assert temps_used == [0.0]
