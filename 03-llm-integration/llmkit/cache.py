"""Response caching for LLM calls.

THE CORRECTNESS CONSTRAINT NOBODY MENTIONS

Caching an LLM response is only sound at `temperature=0`.

At temperature > 0 the model is *supposed* to return different text for the same
input -- that is what the parameter means. Cache it and you have replaced a
sampling distribution with a single frozen draw, silently. The user asking
"give me another variation" gets the same paragraph forever, and it looks like a
product bug rather than a caching bug, so it takes weeks to find.

This module refuses to cache non-deterministic requests by default. That is a
feature. Override it only when you genuinely want "first answer wins" semantics
and have written down why.

WHAT A CACHE IS WORTH HERE

Unlike a DB query cache (saves milliseconds), an LLM cache saves ~2-20 seconds
*and* real money. On workloads with repeated prompts -- FAQ bots, classification
over duplicate records, retry storms, CI runs -- hit rates of 30-60% are normal.
That is a straight 30-60% cut to both latency and spend.

The Java analogue is Caffeine: bounded size, TTL, and eviction on both.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .types import LLMResponse, Message, Usage

# Default TTL. LLM answers are not eternally valid: prompts get edited, models
# get deprecated, and RAG context goes stale. An unbounded cache is a correctness
# hazard as well as a memory leak.
DEFAULT_TTL_SECONDS = 3600.0


def make_cache_key(
    messages: Sequence[Message],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Stable hash over everything that can change the response.

    Every parameter that affects output MUST be in the key. Omitting one is a
    cache-poisoning bug: a `max_tokens=50` request would serve a cached
    `max_tokens=4000` answer.

    `sort_keys=True` matters -- dict iteration order must not change the key, or
    the same logical request hashes differently across processes and the hit
    rate silently collapses to zero.

    >>> k1 = make_cache_key([Message("user", "hi")], "gpt-4o-mini")
    >>> k2 = make_cache_key([Message("user", "hi")], "gpt-4o-mini")
    >>> k1 == k2
    True
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [[m.role, m.content] for m in messages],
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_cacheable(temperature: float) -> bool:
    """Only deterministic requests may be cached. See the module docstring."""
    return temperature == 0.0


@dataclass
class _Entry:
    response: LLMResponse
    expires_at: float


class LRUCache:
    """In-memory LRU cache with TTL, bounded in both size and age.

    Thread-safe: uvicorn and most Python web servers run handlers on multiple
    threads, and `OrderedDict.move_to_end` during another thread's `popitem` is
    a genuine race.

    Single-process only. Across replicas, each pod keeps its own cache, so your
    real hit rate is roughly (measured hit rate / replica count). Use RedisCache
    when that matters.
    """

    def __init__(self, max_size: int = 1000, ttl: float = DEFAULT_TTL_SECONDS) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, key: str, *, now: Optional[float] = None) -> Optional[LLMResponse]:
        now = time.monotonic() if now is None else now
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                # Lazy expiry: cheaper than a background sweeper and adequate
                # because entries are also bounded by max_size.
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            # cached=True lets callers distinguish a served-from-cache response
            # in logs and metrics. Without it, cache hit rate is unobservable.
            return _mark_cached(entry.response)

    def set(self, key: str, response: LLMResponse, *, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._data[key] = _Entry(response, now + self.ttl)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)  # evict least-recently-used
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _mark_cached(response: LLMResponse) -> LLMResponse:
    """Return a copy flagged as cached, with usage zeroed for billing.

    Usage is zeroed deliberately: a cache hit made no provider call, so counting
    its tokens again would double-count spend and make your cost dashboard
    disagree with the provider invoice. The original token counts stay available
    on the stored entry.
    """
    return LLMResponse(
        text=response.text,
        usage=Usage(0, 0),
        model=response.model,
        finish_reason=response.finish_reason,
        latency_ms=0.0,
        cached=True,
        tool_calls=response.tool_calls,
        cost_usd=0.0,
    )


class RedisCache:
    """Shared cache across replicas, backed by Redis.

    Optional dependency: importing this module never requires `redis`; the
    client is imported lazily in the constructor so the rest of the cache module
    works without it.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        ttl: float = DEFAULT_TTL_SECONDS,
        prefix: str = "llmkit:",
        client: Optional[Any] = None,
    ) -> None:
        if client is None:
            try:
                import redis  # optional dependency
            except ImportError as exc:  # pragma: no cover - exercised only without redis
                raise ImportError(
                    "RedisCache needs the `redis` package: pip install redis"
                ) from exc
            client = redis.Redis.from_url(url, decode_responses=True)
        self._client = client
        self.ttl = ttl
        self.prefix = prefix
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, key: str) -> Optional[LLMResponse]:
        raw = self._client.get(self.prefix + key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        data = json.loads(raw)
        return LLMResponse(
            text=data["text"],
            usage=Usage(0, 0),
            model=data["model"],
            finish_reason=data.get("finish_reason", "stop"),
            cached=True,
        )

    def set(self, key: str, response: LLMResponse) -> None:
        payload = json.dumps(
            {
                "text": response.text,
                "model": response.model,
                "finish_reason": response.finish_reason,
            }
        )
        # setex, not set+expire: one round trip, and no window where a crash
        # between the two commands leaves an immortal key.
        self._client.setex(self.prefix + key, int(self.ttl), payload)

    def clear(self) -> None:
        for key in self._client.scan_iter(self.prefix + "*"):
            self._client.delete(key)
