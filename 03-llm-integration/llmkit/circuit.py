"""Circuit breaker for LLM provider calls.

WHY RETRY ALONE IS NOT ENOUGH

Retry handles a *blip*. It makes an outage worse.

When a provider is genuinely down, every request retries 3 times before failing.
You have just tripled the load on a service that is already failing, tripled your
own latency (each request now takes the full backoff budget before erroring), and
tied up three times as many workers waiting. That is a retry storm, and it is how
a partial provider degradation becomes a total outage of *your* service.

A circuit breaker adds the missing rule: after enough consecutive failures, stop
calling and fail immediately. Fast failure frees your workers, sheds load from
the struggling provider, and lets you serve a fallback instead of a timeout.

    CLOSED  --failures >= threshold-->  OPEN
    OPEN    --after recovery_timeout-->  HALF_OPEN
    HALF_OPEN --success x success_threshold--> CLOSED
    HALF_OPEN --any failure-->            OPEN   (restart the timer)

HALF_OPEN is the subtle state: it admits a *small number* of trial requests to
test recovery. Without it you either stay open forever or slam the provider with
full traffic the instant the timer expires.

This is Resilience4j's CircuitBreaker, and the same vocabulary
(CLOSED/OPEN/HALF_OPEN, failure threshold, wait duration) carries over exactly.

WHAT COUNTS AS A FAILURE

Only errors that indicate the *provider* is unhealthy: 5xx, timeouts, connection
errors. A 400 or a 401 must NOT open the circuit -- those are your bugs, and
tripping the breaker on them takes down a working provider because one caller
sent a malformed request. This is the single most common breaker
misconfiguration.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable
from enum import Enum
from typing import Any, Callable, TypeVar

from .errors import CircuitOpenError, LLMError

T = TypeVar("T")


class State(str, Enum):
    CLOSED = "closed"  # normal operation
    OPEN = "open"  # failing fast
    HALF_OPEN = "half_open"  # probing for recovery


def default_is_failure(exc: BaseException) -> bool:
    """Only provider-health signals trip the breaker.

    `LLMError.retryable` already encodes exactly this distinction (5xx/429/
    timeouts are retryable; 4xx client errors are not), so we reuse it rather
    than maintaining a second, divergent list.
    """
    if isinstance(exc, LLMError):
        return exc.retryable
    # Unknown exception types are assumed to be genuine failures.
    return True


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Args:
        failure_threshold: consecutive failures in CLOSED before opening.
        recovery_timeout: seconds to stay OPEN before probing.
        success_threshold: consecutive successes in HALF_OPEN before closing.
        half_open_max_calls: concurrent probes allowed in HALF_OPEN. Keeping
            this small is the point of the state -- a recovering provider must
            not be hit with full traffic.
        is_failure: predicate deciding whether an exception counts against us.
        clock: injectable time source; tests must not sleep.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 1,
        is_failure: Callable[[BaseException], bool] = default_is_failure,
        clock: Callable[[], float] = time.monotonic,
        name: str = "llm",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self._is_failure = is_failure
        self._clock = clock
        self.name = name

        self._state = State.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._lock = threading.RLock()

        # Counters worth exporting as metrics: a breaker you cannot observe is
        # a breaker you will not notice tripping.
        self.rejected_calls = 0
        self.state_changes = 0

    # -- state ------------------------------------------------------------

    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        """Transition OPEN -> HALF_OPEN once the recovery timer has elapsed."""
        if self._state is State.OPEN and self._clock() - self._opened_at >= self.recovery_timeout:
            self._transition(State.HALF_OPEN)
            self._half_open_calls = 0
            self._successes = 0

    def _transition(self, new_state: State) -> None:
        if new_state is not self._state:
            self._state = new_state
            self.state_changes += 1

    def _open(self) -> None:
        self._transition(State.OPEN)
        self._opened_at = self._clock()
        self._failures = 0
        self._successes = 0

    # -- outcome recording -------------------------------------------------

    def _allow(self) -> None:
        """Raise if the call must not proceed."""
        with self._lock:
            self._maybe_half_open()
            if self._state is State.OPEN:
                self.rejected_calls += 1
                elapsed = self._clock() - self._opened_at
                raise CircuitOpenError(
                    f"circuit '{self.name}' is open; failing fast",
                    retry_after=max(0.0, self.recovery_timeout - elapsed),
                )
            if self._state is State.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self.rejected_calls += 1
                    raise CircuitOpenError(
                        f"circuit '{self.name}' is half-open and already probing"
                    )
                self._half_open_calls += 1

    def record_success(self) -> None:
        with self._lock:
            if self._state is State.HALF_OPEN:
                self._successes += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)
                if self._successes >= self.success_threshold:
                    self._transition(State.CLOSED)
                    self._failures = 0
                    self._successes = 0
            else:
                # Consecutive-failure semantics: one success clears the streak.
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state is State.HALF_OPEN:
                # A failure while probing means the provider is still sick.
                # Re-open immediately and restart the full timer.
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._open()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open()

    def reset(self) -> None:
        with self._lock:
            self._state = State.CLOSED
            self._failures = 0
            self._successes = 0
            self._half_open_calls = 0

    # -- invocation --------------------------------------------------------

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self._allow()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised after recording
            if self._is_failure(exc):
                self.record_failure()
            else:
                # Client errors must not count against provider health, but they
                # must also not be mistaken for a success that clears the streak.
                with self._lock:
                    if self._state is State.HALF_OPEN:
                        self._half_open_calls = max(0, self._half_open_calls - 1)
            raise
        self.record_success()
        return result

    async def call_async(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        self._allow()
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised after recording
            if self._is_failure(exc):
                self.record_failure()
            else:
                with self._lock:
                    if self._state is State.HALF_OPEN:
                        self._half_open_calls = max(0, self._half_open_calls - 1)
            raise
        self.record_success()
        return result

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
            f"failures={self._failures}, rejected={self.rejected_calls})"
        )


class _ManualClock:
    """Test helper: advance time without sleeping."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def manual_clock(now: float = 0.0) -> _ManualClock:
    """Return an injectable clock so breaker tests run instantly.

    >>> clock = manual_clock()
    >>> cb = CircuitBreaker(recovery_timeout=30, clock=clock)
    >>> clock.advance(31)
    """
    return _ManualClock(now)


def make_breaker(**kwargs: Any) -> CircuitBreaker:
    """Convenience factory used by the client façade."""
    return CircuitBreaker(**kwargs)


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "State",
    "default_is_failure",
    "make_breaker",
    "manual_clock",
]
