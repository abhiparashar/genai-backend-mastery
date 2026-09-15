"""Cross-cutting request handling: correlation ids, logging, rate limiting.

These are Servlet Filters. FastAPI middleware wraps every request the same
way, in registration order, outermost first.

RATE LIMITING: WHY TOKEN BUCKET

Four common algorithms, and the reason to pick this one:

    fixed window     simple, but allows 2x the limit across a boundary --
                     60 requests at 11:59:59 and 60 more at 12:00:00
    sliding log      exact, but stores every timestamp: O(N) memory per key
    leaky bucket     smooth output, no burst tolerance at all
    TOKEN BUCKET     tolerates a burst up to the bucket size, then enforces a
                     steady refill rate. O(1) memory, two floats per key.

Burst tolerance is the deciding feature. Real clients are bursty -- a page
loads and fires six requests at once. A limiter that rejects that is
technically correct and practically useless.

    tokens = min(capacity, tokens + elapsed * refill_rate)

No background timer: tokens are computed lazily from elapsed time whenever
the key is touched. That is what makes it O(1) and trivially correct.

SINGLE PROCESS ONLY. Each replica keeps its own bucket, so N replicas allow
N times the limit. Production needs Redis (`INCR` + `EXPIRE`, or a Lua script
for exactness). The class below is honest about that rather than pretending.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .observability import (
    METRICS,
    correlation_id_var,
    new_correlation_id,
    tenant_var,
)

logger = logging.getLogger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"


@dataclass
class TokenBucket:
    capacity: float
    refill_per_second: float
    tokens: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)

    def consume(self, amount: float = 1.0, *, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        # Clamp at zero. Time going backwards must never DRAIN a bucket:
        # a negative elapsed would subtract tokens and throttle a client for
        # something the clock did. Monotonic clocks should not regress, but
        # an injected test clock or a rewritten `now` can, and the failure
        # mode (mysterious 429s) is expensive to diagnose.
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self, amount: float = 1.0) -> float:
        """Seconds until `amount` tokens are available. Sent as `Retry-After`.

        Telling a client exactly when to come back is the difference between a
        polite backoff and a retry storm.
        """
        if self.tokens >= amount or self.refill_per_second <= 0:
            return 0.0
        return (amount - self.tokens) / self.refill_per_second


class RateLimiter:
    """Per-key token buckets."""

    def __init__(self, per_minute: int = 60, burst: int = 10) -> None:
        self.per_minute = per_minute
        self.burst = burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                # Start full, so a client's first request is never rejected.
                bucket = TokenBucket(
                    capacity=float(self.burst),
                    refill_per_second=self.per_minute / 60.0,
                    tokens=float(self.burst),
                )
                self._buckets[key] = bucket
            return bucket

    def check(self, key: str) -> tuple[bool, float, float]:
        """Returns (allowed, tokens_remaining, retry_after_seconds)."""
        bucket = self._bucket(key)
        with self._lock:
            allowed = bucket.consume()
            return allowed, max(0.0, bucket.tokens), bucket.retry_after()

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to the request context and the response.

    Accepts an inbound id so a trace started in the Java gateway (module 07)
    continues here instead of restarting. That continuity is the entire point.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get(CORRELATION_HEADER)
        correlation_id = incoming or new_correlation_id()
        correlation_id_var.set(correlation_id)
        tenant_var.set(request.headers.get("X-Tenant-ID", "-"))

        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = correlation_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """One structured line per request, plus latency metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        started = time.monotonic()
        route = request.url.path

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - started) * 1000.0
            METRICS.record_request(route, 500, duration_ms)
            # exc_info so the traceback reaches the logs; the CLIENT still
            # gets a generic message from the exception handler.
            logger.exception("unhandled error on %s after %.0fms", route, duration_ms)
            raise

        duration_ms = (time.monotonic() - started) * 1000.0
        METRICS.record_request(route, response.status_code, duration_ms)
        logger.info(
            "%s %s -> %d in %.0fms",
            request.method,
            route,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.0f}"
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies using Content-Length, before reading them.

    Checking the header first means a 500MB upload is refused without being
    buffered into memory. A body-size limit that reads the body has already
    lost.
    """

    def __init__(self, app: Any, max_bytes: int = 256_000) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": f"request body exceeds {self.max_bytes} bytes",
                        "correlation_id": correlation_id_var.get(),
                    }
                },
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-API-key token bucket, with the standard headers.

    `X-RateLimit-*` and `Retry-After` are not decoration: they are how a
    well-behaved client avoids hammering you. A bare 429 teaches clients
    nothing and produces retry storms.
    """

    def __init__(self, app: Any, limiter: RateLimiter, exempt: Optional[set] = None) -> None:
        super().__init__(app)
        self.limiter = limiter
        # Health and metrics must never be rate limited: a limiter that blocks
        # the readiness probe will take the pod out of service under load,
        # which is precisely backwards.
        self.exempt = exempt or {"/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exempt:
            return await call_next(request)

        key = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "anon"
        )
        allowed, remaining, retry_after = self.limiter.check(key)

        if not allowed:
            METRICS.rate_limited_total += 1
            logger.warning("rate limited %s on %s", key[:8], request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": f"rate limit exceeded; retry in {retry_after:.1f}s",
                        "correlation_id": correlation_id_var.get(),
                    }
                },
                headers={
                    "Retry-After": str(max(1, int(retry_after + 0.999))),
                    "X-RateLimit-Limit": str(self.limiter.per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        return response
