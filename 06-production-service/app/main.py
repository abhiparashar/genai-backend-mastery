"""Application factory and wiring.

The Spring Boot `@SpringBootApplication` equivalent: build the app, register
middleware and routers, install exception handlers, and manage lifecycle.

MIDDLEWARE ORDER IS LOAD-BEARING

Starlette runs middleware in REVERSE registration order for the request path
(last registered is outermost). The order below is chosen so that:

- correlation id is set FIRST, so every later log line and error carries it
- body size is rejected before anything reads the body
- rate limiting happens before auth, so an unauthenticated flood is cheap to
  refuse -- checking a token bucket costs less than a credential comparison

Get this backwards and you do expensive work for requests you were going to
reject anyway.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .deps import get_budgets
from .middleware import (
    BodySizeLimitMiddleware,
    CorrelationIdMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from .observability import configure_logging, correlation_id_var
from .routers import chat, health

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str) -> JSONResponse:
    """Every non-2xx uses one envelope, so clients write one error handler."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "correlation_id": correlation_id_var.get(),
            }
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown. The `@PostConstruct` / `@PreDestroy` equivalent."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    get_budgets().configure(
        global_limit_usd=settings.daily_budget_usd,
        tenant_limit_usd=settings.tenant_daily_budget_usd,
    )

    logger.info(
        "starting %s env=%s provider=%s model=%s",
        settings.app_name,
        settings.environment,
        settings.llm_provider,
        settings.llm_model,
    )
    # Shout about dangerous configuration, but still start: refusing to boot a
    # dev box over a warning is its own kind of unhelpful.
    for warning in settings.startup_check():
        logger.warning("startup check: %s", warning)

    yield

    logger.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="GenAI Service",
        version="1.0.0",
        description=(
            "Reference production LLM service: SSE streaming, API-key auth, "
            "token-bucket rate limiting, per-tenant cost quotas, PII-redacted "
            "structured logs, and Prometheus metrics. Runs offline with no keys."
        ),
        lifespan=lifespan,
    )

    # Registered outermost-last. See the module docstring.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-RateLimit-Remaining", "Retry-After"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        limiter=RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst),
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)  # outermost: runs first

    app.include_router(health.router)
    app.include_router(chat.router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field-level detail is safe and genuinely useful to a caller.
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        return _error(422, "invalid_request", f"{location}: {first.get('msg', 'invalid request')}")

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            return _error(
                exc.status_code,
                detail.get("code", "invalid_request"),
                detail.get("message", "request failed"),
            )
        return _error(exc.status_code, "invalid_request", str(detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the logs, keyed by correlation id. The CLIENT
        # gets a generic message -- stack traces in an HTTP response leak
        # file paths, library versions, and sometimes credentials.
        logger.exception("unhandled exception on %s", request.url.path)
        return _error(500, "internal", "an internal error occurred")

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/healthz",
            "metrics": "/metrics",
        }

    return app


app = create_app()
