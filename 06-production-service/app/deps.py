"""Dependency injection.

`Depends()` is `@Autowired`, with two differences worth knowing: it is
per-request rather than per-singleton by default, and it composes -- a
dependency can depend on other dependencies, and FastAPI resolves the graph.

Anything declared here also appears in the OpenAPI schema, so security
requirements are documented automatically rather than in a wiki nobody reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request

from .config import Settings, get_settings
from .costs import BudgetRegistry
from .llm import build_provider
from .observability import correlation_id_var
from .security import verify_api_key

logger = logging.getLogger(__name__)

# Process-wide singletons. Created once at import, shared across requests --
# the Spring `@Bean` equivalent. Handing these out through Depends() rather
# than importing them directly is what makes them swappable in tests.
_BUDGETS = BudgetRegistry()
_PROVIDER: Optional[Any] = None


def get_budgets() -> BudgetRegistry:
    return _BUDGETS


def get_provider(settings: Settings = Depends(get_settings)) -> Any:
    """Lazily build and cache the provider.

    Built on first use rather than at import so that tests can set environment
    variables before anything reads them.
    """
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = build_provider(settings)
        logger.info("llm provider initialised: %s", type(_PROVIDER).__name__)
    return _PROVIDER


def reset_provider() -> None:
    """Test hook: force the next request to rebuild the provider."""
    global _PROVIDER
    _PROVIDER = None


def set_provider(provider: Any) -> None:
    """Test hook: inject a scripted provider."""
    global _PROVIDER
    _PROVIDER = provider


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Tenant drives cost attribution and isolation."""

    api_key: str
    tenant: str

    @property
    def masked_key(self) -> str:
        """Never log a full key, even at DEBUG."""
        return f"{self.api_key[:4]}...{self.api_key[-2:]}" if len(self.api_key) > 6 else "***"


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """Authenticate the request, or raise 401.

    Returns a `Principal` rather than a bool so downstream code has the tenant
    without re-parsing headers. The 401 body carries no hint about WHY the key
    failed -- "unknown key" and "revoked key" are the same message, because
    the difference is only useful to an attacker.
    """
    if not verify_api_key(x_api_key, tuple(settings.valid_api_keys)):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "unauthorized",
                "message": "a valid X-API-Key header is required",
                "correlation_id": correlation_id_var.get(),
            },
        )
    # Default the tenant to the key itself, so cost is always attributable
    # even when a caller omits the header.
    return Principal(api_key=x_api_key or "", tenant=x_tenant_id or (x_api_key or "unknown"))


async def get_correlation(request: Request) -> str:
    return request.headers.get("X-Correlation-ID") or correlation_id_var.get()
