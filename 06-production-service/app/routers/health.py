"""Health, readiness, and metrics. The Actuator equivalent.

LIVENESS VS READINESS -- THEY ARE NOT THE SAME CHECK

Getting this wrong causes outages, so it is worth being precise:

    /healthz  LIVENESS  "is this process alive?"
              Must NOT check dependencies. If it does, a brief Redis blip
              fails the liveness probe, Kubernetes RESTARTS every pod at
              once, and a degraded service becomes a dead one.

    /readyz   READINESS "can this process serve traffic right now?"
              SHOULD check dependencies. Failing removes the pod from the
              load balancer without killing it, so it recovers on its own
              when the dependency returns.

The rule: liveness answers "restart me?", readiness answers "route to me?".
A dependency check belongs only in the second.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response

from ..config import Settings, get_settings
from ..deps import get_budgets, get_provider
from ..observability import METRICS
from ..schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["ops"])

STARTED_AT = time.monotonic()


@router.get("/healthz", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness. Deliberately trivial: no I/O, no dependencies, always fast."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse)
async def readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
    provider=Depends(get_provider),
) -> ReadyResponse:
    """Readiness. Reports 503 when a dependency makes us unable to serve."""
    checks = {
        "provider": type(provider).__name__,
        "provider_configured": settings.llm_provider == "fake" or bool(settings.provider_key()),
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 1),
    }

    budgets = get_budgets()
    remaining = budgets.remaining("-")["global_remaining_usd"]
    checks["budget_remaining_usd"] = remaining
    # An exhausted budget means we cannot serve, so stop taking traffic. Being
    # removed from the load balancer is the correct response to "I would fail
    # every request anyway".
    checks["budget_ok"] = remaining > 0

    healthy = bool(checks["provider_configured"]) and bool(checks["budget_ok"])
    if not healthy:
        response.status_code = 503
    return ReadyResponse(status="ready" if healthy else "degraded", checks=checks)


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus text exposition.

    Plain text, not JSON: Prometheus scrapes this format, and the content type
    matters to the scraper.
    """
    return Response(content=METRICS.prometheus(), media_type="text/plain; version=0.0.4")


@router.get("/stats")
async def stats() -> dict:
    """Human-readable snapshot. Handy in development; not a Prometheus target."""
    return METRICS.snapshot()
