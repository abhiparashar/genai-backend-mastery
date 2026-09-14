"""Retry with exponential backoff and full jitter.

Implemented by hand so you can see the math. In production you would reach for
`tenacity` (or Resilience4j's `Retry` in Java) -- the logic below is exactly what
those libraries do, and the README shows the tenacity equivalent side by side.

THE THREE RULES

1. Retry only what can succeed.
   Driven entirely by `LLMError.retryable`. A 401 retried three times is three
   guaranteed failures, 3x the latency, and zero chance of success.

2. Back off exponentially, then randomize -- "full jitter".
       sleep = random_uniform(0, min(cap, base * 2**attempt))
   Without the random term, N clients that failed together retry together, and
   keep colliding forever. This is the thundering herd, and it is why naive
   `sleep(2 ** attempt)` makes an overloaded provider worse. AWS's
   "Exponential Backoff and Jitter" post is the canonical reference; full jitter
   wins on both completion time and server load.

3. Obey Retry-After when the provider sends it.
   The provider knows exactly when its window resets. Your curve is a guess.

Also bounded by wall clock (`max_elapsed`), because 5 retries of a 30s timeout is
a 2.5-minute request that some upstream gave up on long ago.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from .errors import LLMError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Knobs for the backoff curve.

    Defaults are tuned for LLM APIs: they are slow and rate limits are common, so
    we are patient (up to ~60s total) but never unbounded.
    """

    max_attempts: int = 3  # total tries, NOT retries-after-the-first
    base_delay: float = 0.5  # seconds; first backoff window
    max_delay: float = 20.0  # per-sleep ceiling
    max_elapsed: float = 60.0  # wall-clock budget across all attempts
    jitter: bool = True  # full jitter; disable only in tests
    respect_retry_after: bool = True
    # Safe as lowercase builtins on 3.9: `from __future__ import annotations` makes
    # dataclass annotations lazy strings, and the default value is a real tuple.
    retry_on: tuple[type[BaseException], ...] = (LLMError,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be > 0")


def compute_delay(
    attempt: int, policy: RetryPolicy, *, retry_after: Optional[float] = None
) -> float:
    """Delay before the next attempt. `attempt` is 0-based (0 = after first failure).

    >>> p = RetryPolicy(base_delay=1.0, max_delay=20.0, jitter=False)
    >>> [compute_delay(i, p) for i in range(5)]
    [1.0, 2.0, 4.0, 8.0, 16.0]

    With jitter the value is uniform in [0, capped]; the ceiling is unchanged, so
    the expected wait halves while the worst case stays bounded.
    """
    if retry_after is not None and policy.respect_retry_after:
        # Still capped: a hostile or buggy Retry-After of 3600 must not hang us.
        return min(float(retry_after), policy.max_delay)

    capped = min(policy.max_delay, policy.base_delay * (2**attempt))
    if policy.jitter:
        return random.uniform(0.0, capped)
    return capped


def _should_retry(exc: BaseException, policy: RetryPolicy) -> bool:
    if not isinstance(exc, policy.retry_on):
        return False
    # LLMError carries its own verdict; anything else in retry_on is opt-in.
    if isinstance(exc, LLMError):
        return exc.retryable
    return True


def _retry_after_of(exc: BaseException) -> Optional[float]:
    if isinstance(exc, RateLimitError):
        return exc.retry_after
    return None


def with_retry(
    fn: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """Call `fn` with retries. `sleep` is injectable so tests run instantly."""
    policy = policy or RetryPolicy()
    started = time.monotonic()
    last_exc: Optional[BaseException] = None

    for attempt in range(policy.max_attempts):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last_exc = exc
            if not _should_retry(exc, policy):
                raise
            if attempt == policy.max_attempts - 1:
                raise

            delay = compute_delay(attempt, policy, retry_after=_retry_after_of(exc))
            elapsed = time.monotonic() - started
            if elapsed + delay > policy.max_elapsed:
                logger.warning(
                    "retry budget exhausted after %.1fs (attempt %d/%d)",
                    elapsed,
                    attempt + 1,
                    policy.max_attempts,
                )
                raise

            logger.info(
                "attempt %d/%d failed (%s); retrying in %.2fs",
                attempt + 1,
                policy.max_attempts,
                type(exc).__name__,
                delay,
            )
            sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_exc  # type: ignore[misc]


async def with_retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    **kwargs: Any,
) -> T:
    """Async twin of `with_retry`.

    Note `asyncio.sleep` rather than `time.sleep`: blocking the event loop during
    a backoff would stall every other in-flight request in the process. This is
    the most common async retry bug.
    """
    policy = policy or RetryPolicy()
    do_sleep = sleep or asyncio.sleep
    started = time.monotonic()

    for attempt in range(policy.max_attempts):
        try:
            return await fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if not _should_retry(exc, policy):
                raise
            if attempt == policy.max_attempts - 1:
                raise

            delay = compute_delay(attempt, policy, retry_after=_retry_after_of(exc))
            if (time.monotonic() - started) + delay > policy.max_elapsed:
                raise
            await do_sleep(delay)

    raise AssertionError("unreachable")
