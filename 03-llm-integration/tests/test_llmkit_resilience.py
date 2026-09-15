"""Retry, backoff, and circuit-breaker behaviour.

These assert the decisions that cost money or cause outages when wrong:
what gets retried, how long we wait, and when we stop calling entirely.
No test here sleeps -- both the clock and the sleep function are injected.
"""

from __future__ import annotations

import pytest
from llmkit.circuit import CircuitBreaker, State, manual_clock
from llmkit.errors import (
    AuthError,
    CircuitOpenError,
    ContextLengthError,
    InvalidRequestError,
    RateLimitError,
    TransientError,
    classify_status,
)
from llmkit.providers import FakeProvider
from llmkit.retry import RetryPolicy, compute_delay, with_retry, with_retry_async
from llmkit.types import user

# ---------------------------------------------------------------------------
# Error classification -- the retryable/fatal split is the core of the library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_type", "retryable"),
    [
        (429, RateLimitError, True),
        (500, TransientError, True),
        (502, TransientError, True),
        (503, TransientError, True),
        (401, AuthError, False),
        (403, AuthError, False),
        (400, InvalidRequestError, False),
        (422, InvalidRequestError, False),
    ],
)
def test_status_maps_to_error_with_correct_retryability(status, expected_type, retryable):
    error = classify_status(status, "body")
    assert isinstance(error, expected_type)
    assert error.retryable is retryable


def test_context_length_is_detected_from_body_not_status():
    # A 400 is normally fatal-and-generic, but this specific one is recoverable
    # by shrinking the input, so callers need to distinguish it.
    error = classify_status(400, "This model's maximum context length is 8192 tokens")
    assert isinstance(error, ContextLengthError)
    assert error.retryable is False


def test_retry_after_is_captured_from_the_header():
    error = classify_status(429, "slow down", retry_after=12.0)
    assert isinstance(error, RateLimitError)
    assert error.retry_after == 12.0


# ---------------------------------------------------------------------------
# Backoff maths
# ---------------------------------------------------------------------------


def test_backoff_doubles_and_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=False)
    assert [compute_delay(i, policy) for i in range(6)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_full_jitter_stays_within_the_exponential_ceiling():
    # Full jitter picks uniformly in [0, ceiling]. The ceiling must still hold,
    # otherwise a "jittered" backoff could sleep longer than max_delay.
    policy = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=True)
    samples = [compute_delay(3, policy) for _ in range(500)]
    assert all(0.0 <= s <= 8.0 for s in samples)
    assert len(set(samples)) > 1, "jitter must actually randomise"


def test_retry_after_overrides_the_curve_but_is_still_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=20.0, jitter=False)
    assert compute_delay(0, policy, retry_after=7.0) == 7.0
    # A hostile or buggy Retry-After must not hang the caller for an hour.
    assert compute_delay(0, policy, retry_after=3600.0) == 20.0


# ---------------------------------------------------------------------------
# Retry driver
# ---------------------------------------------------------------------------


def test_retries_transient_failures_then_succeeds():
    provider = FakeProvider(
        responses=["recovered"], fail_times=2, failure=RateLimitError("429", retry_after=0)
    )
    result = with_retry(
        lambda: provider.complete([user("hi")]),
        policy=RetryPolicy(max_attempts=5),
        sleep=lambda _: None,
    )
    assert result.text == "recovered"
    assert provider.calls == 3


def test_fatal_errors_are_not_retried():
    calls = {"n": 0}

    def always_401():
        calls["n"] += 1
        raise AuthError("bad key")

    with pytest.raises(AuthError):
        with_retry(always_401, policy=RetryPolicy(max_attempts=5), sleep=lambda _: None)
    assert calls["n"] == 1, "a 401 will not become valid on retry"


def test_gives_up_after_max_attempts():
    provider = FakeProvider(fail_times=99, failure=TransientError("503"))
    with pytest.raises(TransientError):
        with_retry(
            lambda: provider.complete([user("hi")]),
            policy=RetryPolicy(max_attempts=3),
            sleep=lambda _: None,
        )
    assert provider.calls == 3


def test_wall_clock_budget_stops_retrying_early():
    # max_elapsed must win over max_attempts: 10 attempts of a slow call is a
    # request some upstream abandoned long ago.
    policy = RetryPolicy(max_attempts=10, base_delay=5.0, max_elapsed=6.0, jitter=False)
    provider = FakeProvider(fail_times=99, failure=TransientError("503"))
    slept: list[float] = []
    with pytest.raises(TransientError):
        with_retry(lambda: provider.complete([user("hi")]), policy=policy, sleep=slept.append)
    assert provider.calls < 10
    assert sum(slept) <= 6.0


@pytest.mark.asyncio
async def test_async_retry_recovers():
    provider = FakeProvider(responses=["ok"], fail_times=1, failure=TransientError("503"))

    async def call():
        return await provider.acomplete([user("hi")])

    async def no_sleep(_: float) -> None:
        return None

    result = await with_retry_async(call, policy=RetryPolicy(max_attempts=3), sleep=no_sleep)
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def _boom():
    raise TransientError("503")


def test_breaker_opens_after_threshold_then_fails_fast():
    breaker = CircuitBreaker(failure_threshold=3, clock=manual_clock())
    for _ in range(3):
        with pytest.raises(TransientError):
            breaker.call(_boom)

    assert breaker.state is State.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "never runs")
    assert breaker.rejected_calls == 1


def test_breaker_probes_then_closes_after_recovery():
    clock = manual_clock()
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=30, success_threshold=2, clock=clock
    )
    with pytest.raises(TransientError):
        breaker.call(_boom)
    assert breaker.state is State.OPEN

    clock.advance(31)
    assert breaker.state is State.HALF_OPEN

    breaker.call(lambda: "ok")
    assert breaker.state is State.HALF_OPEN, "one success is not enough to trust it"
    breaker.call(lambda: "ok")
    assert breaker.state is State.CLOSED


def test_failure_while_probing_reopens_and_restarts_the_timer():
    clock = manual_clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10, clock=clock)
    with pytest.raises(TransientError):
        breaker.call(_boom)
    clock.advance(11)
    assert breaker.state is State.HALF_OPEN

    with pytest.raises(TransientError):
        breaker.call(_boom)
    assert breaker.state is State.OPEN
    clock.advance(9)
    assert breaker.state is State.OPEN, "the recovery timer must restart, not resume"


def test_client_errors_never_open_the_breaker():
    # The classic misconfiguration: one caller sends malformed requests and
    # trips the breaker, taking down a perfectly healthy provider for everyone.
    breaker = CircuitBreaker(failure_threshold=2, clock=manual_clock())
    for _ in range(10):
        with pytest.raises(InvalidRequestError):
            breaker.call(lambda: (_ for _ in ()).throw(InvalidRequestError("400")))
    assert breaker.state is State.CLOSED


def test_a_success_clears_a_partial_failure_streak():
    breaker = CircuitBreaker(failure_threshold=3, clock=manual_clock())
    with pytest.raises(TransientError):
        breaker.call(_boom)
    breaker.call(lambda: "ok")
    for _ in range(2):
        with pytest.raises(TransientError):
            breaker.call(_boom)
    assert breaker.state is State.CLOSED, "threshold counts CONSECUTIVE failures"
