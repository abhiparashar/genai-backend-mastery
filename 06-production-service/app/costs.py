"""Per-tenant spend tracking and quota enforcement.

WHY THIS IS AN API CONCERN, NOT A BILLING ONE

In a normal service, a runaway client costs you CPU you already paid for.
Here it costs money per request, with no ceiling. Two distinct limits are
needed, and they fail differently:

    GLOBAL budget   protects YOU. Breached means the service stops spending.
    TENANT quota    protects you from ONE customer. Breached means that
                    customer stops, everyone else continues.

Without the per-tenant limit, one abusive client exhausts the global budget
and takes the service down for every paying customer -- turning a billing
problem into an availability incident.

ENFORCE BEFORE THE CALL. A check afterwards is an invoice.

HTTP SEMANTICS: 402 vs 429

    429 Too Many Requests  -- you are going too fast; slow down and retry
    402 Payment Required   -- you are out of budget; retrying will not help

Returning 429 for an exhausted quota is actively harmful: it tells a
well-behaved client to retry, which it will, forever.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


class BudgetExceeded(Exception):
    """Raised pre-flight when a call would breach a limit."""

    def __init__(self, message: str, *, scope: str, spent: float, limit: float) -> None:
        super().__init__(message)
        self.scope = scope  # "global" | "tenant"
        self.spent = spent
        self.limit = limit


@dataclass
class Window:
    """A spend total that resets on a fixed period."""

    limit_usd: float
    period_seconds: float = 86_400.0
    spent_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def _roll(self, now: float) -> None:
        if now - self.started_at >= self.period_seconds:
            self.spent_usd = 0.0
            self.started_at = now

    def would_exceed(self, projected: float, *, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        self._roll(now)
        return self.spent_usd + projected > self.limit_usd

    def add(self, amount: float, *, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._roll(now)
        self.spent_usd += amount

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)


class BudgetRegistry:
    """Global and per-tenant budgets.

    In-process, and that is a real limitation stated plainly: with N replicas
    each keeps its own counter, so the effective limit is N times what you
    configured. Production needs a shared counter -- Redis `INCRBYFLOAT` with
    an expiry is enough, and the interface here is deliberately small so
    swapping the backend touches one class.
    """

    def __init__(
        self, global_limit_usd: float = 5.0, tenant_limit_usd: float = 1.0
    ) -> None:
        self.global_window = Window(global_limit_usd)
        self.tenant_limit_usd = tenant_limit_usd
        self._tenants: dict = defaultdict(lambda: Window(tenant_limit_usd))
        self._lock = threading.Lock()

    def configure(self, *, global_limit_usd: float, tenant_limit_usd: float) -> None:
        with self._lock:
            self.global_window.limit_usd = global_limit_usd
            self.tenant_limit_usd = tenant_limit_usd
            for window in self._tenants.values():
                window.limit_usd = tenant_limit_usd

    def check(self, tenant: str, projected_usd: float) -> None:
        """Raise `BudgetExceeded` if this call would breach either limit."""
        with self._lock:
            if self.global_window.would_exceed(projected_usd):
                raise BudgetExceeded(
                    "service budget exhausted",
                    scope="global",
                    spent=self.global_window.spent_usd,
                    limit=self.global_window.limit_usd,
                )
            window = self._tenants[tenant]
            if window.would_exceed(projected_usd):
                raise BudgetExceeded(
                    f"quota exhausted for tenant {tenant!r}",
                    scope="tenant",
                    spent=window.spent_usd,
                    limit=window.limit_usd,
                )

    def record(self, tenant: str, cost_usd: float) -> None:
        with self._lock:
            self.global_window.add(cost_usd)
            self._tenants[tenant].add(cost_usd)

    def remaining(self, tenant: str) -> dict:
        with self._lock:
            return {
                "global_remaining_usd": round(self.global_window.remaining_usd, 6),
                "tenant_remaining_usd": round(self._tenants[tenant].remaining_usd, 6),
            }

    def reset(self) -> None:
        with self._lock:
            self.global_window.spent_usd = 0.0
            self._tenants.clear()
