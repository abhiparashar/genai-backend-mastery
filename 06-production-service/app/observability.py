"""Structured logging, PII redaction, and in-process metrics.

WHAT TO MEASURE IN AN LLM SERVICE

Everything you already measure in a Java service -- latency percentiles, error
rate, throughput -- plus two that are specific to this domain and that people
forget until the invoice arrives:

    TOKENS   the unit of work
    DOLLARS  the unit of consequence

Cost is a first-class metric here, not a monthly finance report. If you cannot
answer "what did the last hour cost, and which tenant caused it?" from your
dashboards, you will find out from your bill instead.

WHY p95 AND p99, NEVER THE MEAN

LLM latency distributions are heavily right-skewed: most calls are quick, a
few are enormous. A mean of 900ms can hide a p99 of 30 seconds, and the p99 is
what your users complain about. The mean is the least useful number available.

CORRELATION IDS

One id, generated at the edge, attached to every log line and returned in the
response header. It is what lets you reconstruct a single request across the
Java gateway (module 07) and this Python service. Without it, debugging a
distributed LLM call means grepping two services by timestamp and hoping.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

# ContextVar, not a global: each concurrent request gets its own value even
# though they share a thread and an event loop. A module-level global would
# leak one request's id into another's logs under load -- and only under load,
# which is the worst kind of bug.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")
tenant_var: ContextVar[str] = ContextVar("tenant", default="-")


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# ORDER MATTERS: most specific first. A greedy phone pattern placed early will
# swallow SSNs, IPs and card numbers -- they are all digits with separators.
#
# Note this deliberately does NOT Luhn-check card numbers, unlike
# agentkit/guardrails.py. The trade-off is different by context: there, a
# false positive corrupts data the agent needs, so precision matters. Here the
# output is a log line, so OVER-redacting a 16-digit order number costs you a
# little debuggability while UNDER-redacting a real card is a PCI incident.
# When the two errors are that asymmetric, bias toward the cheap one.
REDACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("API_KEY", r"\b(?:sk|pk|ghp|gho|xox[baprs])-[A-Za-z0-9_-]{8,}"),
    ("BEARER", r"[Bb]earer\s+[A-Za-z0-9._-]{8,}"),
    ("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("IP", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ("CARD", r"\b(?:\d[ -]?){13,19}\b"),
    ("PHONE", r"(?<![\w-])\+?\d[\d\s()-]{7,12}\d(?![\w-])"),
)

_COMPILED = tuple((label, re.compile(pattern)) for label, pattern in REDACTION_PATTERNS)


def redact(text: str) -> str:
    """Strip secrets and PII from anything about to be logged.

    Redact at the LOGGING boundary, not at each call site. Relying on every
    developer to remember is how a customer's email ends up in CloudWatch for
    seven years, discovered during an audit.

    >>> redact("contact bob@example.com with key sk-abcdefgh12345")
    'contact [EMAIL] with key [API_KEY]'
    """
    for label, pattern in _COMPILED:
        text = pattern.sub(f"[{label}]", text)
    return text


class RedactionFilter(logging.Filter):
    """Applies `redact()` to every record passing through the logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


class JSONFormatter(logging.Formatter):
    """One JSON object per line.

    Machine-parseable by default. Human-readable logs are pleasant right up to
    the moment you need to filter 40 million of them by tenant.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        tenant = tenant_var.get()
        if tenant != "-":
            payload["tenant"] = tenant
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JSONFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(RedactionFilter())

    root = logging.getLogger()
    root.handlers.clear()  # replace uvicorn's default handler
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Histogram:
    """Bounded-memory latency histogram.

    A deque with a maxlen keeps the last N observations rather than growing
    forever. Real Prometheus uses fixed buckets; this keeps raw samples so
    percentiles are exact over the window, which is fine at this scale and
    much easier to read.
    """

    maxlen: int = 1000
    values: deque = field(default_factory=lambda: deque(maxlen=1000))

    def observe(self, value: float) -> None:
        self.values.append(value)

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = min(len(ordered) - 1, int(q * len(ordered)))
        return ordered[index]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return sum(self.values)


@dataclass
class Metrics:
    """In-process metrics with a Prometheus text exposition.

    Deliberately dependency-free. In production you would use
    prometheus_client; the point here is to show WHAT to measure, which is the
    part that transfers.

    Lock-guarded because uvicorn serves requests concurrently and `+=` on a
    float is not atomic -- the resulting drift is small, permanent, and
    maddening to explain to finance.
    """

    requests_total: dict = field(default_factory=lambda: defaultdict(int))
    errors_total: dict = field(default_factory=lambda: defaultdict(int))
    latency_ms: Histogram = field(default_factory=Histogram)
    llm_latency_ms: Histogram = field(default_factory=Histogram)
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    cost_usd_total: float = 0.0
    cost_by_tenant: dict = field(default_factory=lambda: defaultdict(float))
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limited_total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self, route: str, status: int, duration_ms: float) -> None:
        with self._lock:
            self.requests_total[(route, status)] += 1
            self.latency_ms.observe(duration_ms)
            if status >= 500:
                self.errors_total[route] += 1

    def record_llm(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        duration_ms: float,
        tenant: str = "-",
    ) -> None:
        with self._lock:
            self.input_tokens_total += input_tokens
            self.output_tokens_total += output_tokens
            self.cost_usd_total += cost_usd
            self.cost_by_tenant[tenant] += cost_usd
            self.llm_latency_ms.observe(duration_ms)

    def record_error(self, route: str) -> None:
        with self._lock:
            self.errors_total[route] += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": sum(self.requests_total.values()),
                "errors": sum(self.errors_total.values()),
                "p50_ms": round(self.latency_ms.percentile(0.50), 1),
                "p95_ms": round(self.latency_ms.percentile(0.95), 1),
                "p99_ms": round(self.latency_ms.percentile(0.99), 1),
                "llm_p95_ms": round(self.llm_latency_ms.percentile(0.95), 1),
                "input_tokens": self.input_tokens_total,
                "output_tokens": self.output_tokens_total,
                "cost_usd": round(self.cost_usd_total, 6),
                "cache_hit_rate": round(self.cache_hit_rate, 3),
                "rate_limited": self.rate_limited_total,
            }

    def prometheus(self) -> str:
        """Prometheus text exposition format.

        The format is plain text: `# HELP`, `# TYPE`, then `name{labels} value`.
        Counters end in `_total` by convention and must be monotonic.
        """
        lines: list = []

        def add(name: str, kind: str, help_text: str, value: Any, labels: str = "") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name}{labels} {value}")

        with self._lock:
            lines.append("# HELP http_requests_total Total HTTP requests")
            lines.append("# TYPE http_requests_total counter")
            for (route, status), count in sorted(self.requests_total.items()):
                lines.append(f'http_requests_total{{route="{route}",status="{status}"}} {count}')

            for quantile, value in (
                ("0.5", self.latency_ms.percentile(0.50)),
                ("0.95", self.latency_ms.percentile(0.95)),
                ("0.99", self.latency_ms.percentile(0.99)),
            ):
                lines.append(f'http_request_duration_ms{{quantile="{quantile}"}} {value:.1f}')

            add("llm_input_tokens_total", "counter", "Prompt tokens", self.input_tokens_total)
            add("llm_output_tokens_total", "counter", "Completion tokens", self.output_tokens_total)
            add("llm_cost_usd_total", "counter", "Spend in USD", f"{self.cost_usd_total:.6f}")
            add("llm_cache_hit_rate", "gauge", "Cache hit ratio", f"{self.cache_hit_rate:.3f}")
            add("http_rate_limited_total", "counter", "429 responses", self.rate_limited_total)

            for tenant, cost in sorted(self.cost_by_tenant.items()):
                lines.append(f'llm_cost_usd_by_tenant{{tenant="{tenant}"}} {cost:.6f}')

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.requests_total.clear()
            self.errors_total.clear()
            self.latency_ms = Histogram()
            self.llm_latency_ms = Histogram()
            self.input_tokens_total = 0
            self.output_tokens_total = 0
            self.cost_usd_total = 0.0
            self.cost_by_tenant.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.rate_limited_total = 0


METRICS = Metrics()


def log_with(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log with structured fields attached to the JSON record."""
    logger.log(level, message, extra={"extra_fields": fields})


def get_correlation_id() -> str:
    return correlation_id_var.get()


def set_correlation_id(value: Optional[str] = None) -> str:
    resolved = value or new_correlation_id()
    correlation_id_var.set(resolved)
    return resolved
