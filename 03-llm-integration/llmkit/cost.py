"""Token counting, price tables, cost attribution, and a pre-flight budget guard.

WHY COST IS AN ENGINEERING CONCERN, NOT A FINANCE ONE

In a normal backend, a bug costs you CPU. In an LLM backend, a bug costs you
money, immediately and without a ceiling. The canonical incident: an agent gets
stuck in a tool-call loop at 2am, makes 40,000 calls before anyone notices, and
turns a $12/day service into a $4,000 invoice. Nothing in the OpenAI dashboard
stops that. Your code has to.

So cost gets the same treatment as latency: measured per request, attributed per
tenant, exported as a metric, and *bounded by a guard that runs BEFORE the call*.
A budget check after the fact is not a guard, it is an invoice.

THREE THINGS THIS MODULE DOES

1. `count_tokens`  - exact via tiktoken when installed, ~4-chars/token otherwise.
2. `estimate_cost` - price table lookup; input and output are priced differently
                     (output is typically 3-5x input, which is why "be concise"
                     in a system prompt is a real cost lever).
3. `BudgetGuard`   - refuses the call when the projected spend breaches a cap.
"""

from __future__ import annotations

import datetime as _dt
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from .errors import BudgetExceededError
from .types import Message, Usage

# ---------------------------------------------------------------------------
# Price table
# ---------------------------------------------------------------------------
#
# !! PRICES CHANGE. This table WILL go stale. !!
#
# Treat it as a default, not a source of truth. In production you load pricing
# from config so it can be corrected without a redeploy, and you alert when a
# model you are calling has no entry (silently costing $0 in your dashboards is
# worse than erroring).
#
# USD per 1,000,000 tokens.
LAST_VERIFIED = _dt.date(2025, 6, 1)

PRICES: dict[str, tuple[float, float]] = {
    # model                      (input_per_1m, output_per_1m)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "o3-mini": (1.10, 4.40),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    # Embeddings: output is always 0, they return vectors not tokens.
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # The offline provider is free, and saying so explicitly keeps test
    # assertions honest instead of silently falling through to a default.
    "fake-1": (0.0, 0.0),
}

# Note the spread: gpt-4o costs ~17x gpt-4o-mini on input and ~17x on output.
# Routing simple requests (classification, extraction, routing itself) to the
# small model is usually the single largest cost win available, and it is a
# config change, not a rewrite. See best-practices/cost-optimization.md.

CHARS_PER_TOKEN = 4


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Token count for `text`.

    Uses tiktoken when available (exact for OpenAI models). Falls back to a
    ~4-chars-per-token heuristic, which is within roughly 10% for English prose
    but degrades badly on code, JSON, and non-Latin scripts -- all of which
    tokenize far less efficiently. Budget accordingly.
    """
    if not text:
        return 0
    try:
        import tiktoken  # optional dependency
    except ImportError:
        return max(1, len(text) // CHARS_PER_TOKEN)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model: cl100k_base is the right default for modern OpenAI models.
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_message_tokens(messages: Sequence[Message], model: str = "gpt-4o-mini") -> int:
    """Token count for a whole conversation.

    Adds ~4 tokens per message of chat-format overhead (role markers and
    delimiters the API inserts). Ignoring this is how people end up 10-15% under
    on long conversations and get surprised by context-length errors.
    """
    return sum(count_tokens(m.content, model) + 4 for m in messages)


def estimate_cost(usage: Usage, model: str) -> float:
    """USD cost for a given usage on a given model. Unknown model -> 0.0.

    >>> round(estimate_cost(Usage(1_000_000, 1_000_000), "gpt-4o-mini"), 4)
    0.75
    """
    if model not in PRICES:
        return 0.0
    input_price, output_price = PRICES[model]
    return (usage.input_tokens / 1_000_000) * input_price + (
        usage.output_tokens / 1_000_000
    ) * output_price


def estimate_request_cost(
    messages: Sequence[Message],
    model: str,
    *,
    expected_output_tokens: int = 500,
) -> float:
    """Project the cost of a call BEFORE making it.

    Output length is unknowable in advance, so we assume `expected_output_tokens`
    (default deliberately generous). A budget guard that under-estimates is not
    a guard.
    """
    return estimate_cost(
        Usage(count_message_tokens(messages, model), expected_output_tokens), model
    )


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    model: str
    usage: Usage
    cost_usd: float
    tenant: str = "default"
    operation: str = "chat"


@dataclass
class CostTracker:
    """Aggregates spend by model and tenant.

    Thread-safe because a FastAPI app under uvicorn runs handlers concurrently
    and `+=` on a float is not atomic across threads. This is the kind of detail
    that produces cost numbers that are quietly ~2% wrong forever.
    """

    entries: list[CostEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        usage: Usage,
        model: str,
        *,
        tenant: str = "default",
        operation: str = "chat",
    ) -> CostEntry:
        entry = CostEntry(model, usage, estimate_cost(usage, model), tenant, operation)
        with self._lock:
            self.entries.append(entry)
        return entry

    @property
    def total_usd(self) -> float:
        with self._lock:
            return sum(e.cost_usd for e in self.entries)

    @property
    def total_usage(self) -> Usage:
        with self._lock:
            total = Usage()
            for e in self.entries:
                total = total + e.usage
            return total

    def by_model(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        with self._lock:
            for e in self.entries:
                out[e.model] += e.cost_usd
        return dict(out)

    def by_tenant(self) -> dict[str, float]:
        """Per-tenant spend. Required for any product that bills usage."""
        out: dict[str, float] = defaultdict(float)
        with self._lock:
            for e in self.entries:
                out[e.tenant] += e.cost_usd
        return dict(out)

    def report(self) -> str:
        usage = self.total_usage
        lines = [
            f"calls={len(self.entries)} "
            f"in={usage.input_tokens} out={usage.output_tokens} "
            f"total=${self.total_usd:.6f}",
        ]
        for model, cost in sorted(self.by_model().items(), key=lambda kv: -kv[1]):
            lines.append(f"  {model:<32} ${cost:.6f}")
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()


class BudgetGuard:
    """Hard spend ceiling, enforced before the request leaves the process.

    Deliberately simple and in-process. In a multi-replica deployment this must
    be backed by a shared counter (Redis `INCRBYFLOAT` with an expiry) or each
    replica gets its own full budget and you overspend by a factor of N. The
    Redis version is in `06-production-service/app/costs.py`.
    """

    def __init__(self, limit_usd: float, *, tracker: Optional[CostTracker] = None) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        self.limit_usd = limit_usd
        self.tracker = tracker or CostTracker()

    @property
    def spent_usd(self) -> float:
        return self.tracker.total_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def check(self, projected_usd: float = 0.0) -> None:
        """Raise `BudgetExceededError` if this call would breach the cap."""
        if self.spent_usd + projected_usd > self.limit_usd:
            raise BudgetExceededError(
                f"budget exceeded: spent ${self.spent_usd:.4f} "
                f"+ projected ${projected_usd:.4f} > limit ${self.limit_usd:.4f}",
                spent_usd=self.spent_usd,
                limit_usd=self.limit_usd,
            )

    def check_request(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        expected_output_tokens: int = 500,
    ) -> None:
        self.check(
            estimate_request_cost(messages, model, expected_output_tokens=expected_output_tokens)
        )

    def record(self, usage: Usage, model: str, **kw: str) -> CostEntry:
        return self.tracker.record(usage, model, **kw)
