"""Exception hierarchy modelling how LLM APIs actually fail.

THIS MODULE IS THE LIBRARY. Everything else is plumbing.

The single most important decision in an LLM client is: *is this error worth
retrying?* Get it wrong in one direction and you hammer a provider with requests
that can never succeed (burning your rate limit and your money). Get it wrong in
the other and a routine 503 becomes a user-visible 500.

    RETRYABLE          -> transient, the same request may succeed later
      RateLimitError     429  (back off, honor Retry-After)
      TransientError     500/502/503/504, timeouts, connection resets
    NOT RETRYABLE      -> the request itself is wrong; retrying is pure waste
      AuthError          401/403  bad or revoked key
      InvalidRequestError 400/422 malformed payload
      ContextLengthError 400 w/ context_length_exceeded -- shrink input instead
      ContentFilterError provider refused on safety grounds
      BudgetExceededError our own guard fired BEFORE the call
"""

from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Base for everything this library raises."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        # Providers return a request id; always log it. It is the only thing
        # their support team can act on.
        self.request_id = request_id

    def __str__(self) -> str:
        bits = [self.message]
        if self.status_code is not None:
            bits.append(f"status={self.status_code}")
        if self.provider:
            bits.append(f"provider={self.provider}")
        if self.request_id:
            bits.append(f"request_id={self.request_id}")
        return " ".join(bits)


class RateLimitError(LLMError):
    """429. Retryable, but ONLY after waiting.

    `retry_after` comes from the provider's Retry-After header. Honoring it beats
    your own backoff curve every time -- the provider knows when the window resets
    and you are guessing.
    """

    retryable = True

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kw: object) -> None:
        super().__init__(message, **kw)  # type: ignore[arg-type]
        self.retry_after = retry_after


class TransientError(LLMError):
    """5xx, timeout, connection reset. Retryable with backoff."""

    retryable = True


class AuthError(LLMError):
    """401/403. Never retry -- your key will not become valid in 2 seconds."""


class InvalidRequestError(LLMError):
    """400/422. Your payload is malformed. Retrying sends the same bad payload."""


class ContextLengthError(InvalidRequestError):
    """Input exceeded the model's context window.

    Not retryable as-is, but *recoverable*: drop old turns, shrink RAG context, or
    route to a longer-context model. Callers should catch this specifically.
    """

    def __init__(
        self,
        message: str,
        *,
        tokens_used: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kw: object,
    ) -> None:
        super().__init__(message, **kw)  # type: ignore[arg-type]
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens


class ContentFilterError(LLMError):
    """Provider refused on safety grounds. Retrying the same prompt is pointless."""


class BudgetExceededError(LLMError):
    """Our own spend guard fired before the request went out.

    Deliberately raised PRE-flight. A post-hoc budget check is an invoice, not a
    guard rail.
    """

    def __init__(self, message: str, *, spent_usd: float = 0.0, limit_usd: float = 0.0) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd


class CircuitOpenError(LLMError):
    """Circuit breaker is open; we are failing fast instead of piling on."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def classify_status(
    status_code: int,
    body: str = "",
    *,
    provider: Optional[str] = None,
    retry_after: Optional[float] = None,
    request_id: Optional[str] = None,
) -> LLMError:
    """Map an HTTP status + body onto the right exception.

    Shared by every HTTP provider so classification lives in exactly one place.
    """
    lowered = body.lower()
    kw = {"status_code": status_code, "provider": provider, "request_id": request_id}

    if status_code == 429:
        return RateLimitError("rate limited", retry_after=retry_after, **kw)  # type: ignore[arg-type]
    if status_code in (401, 403):
        return AuthError("authentication failed; check your API key", **kw)  # type: ignore[arg-type]
    if status_code >= 500:
        return TransientError(f"upstream error {status_code}", **kw)  # type: ignore[arg-type]
    if status_code in (400, 422):
        if (
            "context_length" in lowered
            or "too many tokens" in lowered
            or "maximum context" in lowered
        ):
            return ContextLengthError("input exceeds the model context window", **kw)  # type: ignore[arg-type]
        if "content_filter" in lowered or "content policy" in lowered:
            return ContentFilterError("request blocked by the provider content filter", **kw)  # type: ignore[arg-type]
        return InvalidRequestError(body[:200] or "invalid request", **kw)  # type: ignore[arg-type]
    return LLMError(f"unexpected status {status_code}: {body[:200]}", **kw)  # type: ignore[arg-type]
