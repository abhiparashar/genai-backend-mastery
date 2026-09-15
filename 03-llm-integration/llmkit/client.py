"""`LLMClient` -- the façade your application code actually calls.

Everything else in this package is a single concern: retry, caching, cost,
breaking, structure. This module composes them in the right order, which is
itself a design decision worth stating explicitly:

    budget guard        (cheapest check first -- refuse before spending anything)
      -> cache lookup   (free, instant; skip the network entirely on a hit)
        -> circuit breaker  (fail fast if the provider is known-sick)
          -> retry            (per-attempt, inside the breaker)
            -> provider call
          -> record cost, populate cache

WHY THAT ORDER

- Budget before cache: a cache *hit* costs nothing, so it should not be blocked
  by a budget check... but a budget check is nanoseconds and putting it first
  keeps the "no call proceeds past a breached budget" invariant trivially true.
  We skip the projection cost for cache hits by checking the cache first only
  when the request is cacheable. (See `complete` -- this is the one place the
  ordering bends, deliberately, and it is commented there.)
- Breaker outside retry: retrying *inside* an open circuit defeats the point.
  Conversely, a single failed attempt should not immediately open the breaker
  before retry has had its chance -- so the breaker counts one logical call,
  not one HTTP attempt. Resilience4j composes them the same way.
- Cost recorded after success only. Failed calls that never produced tokens
  cost nothing; failed calls that DID produce tokens are a real (small) leak
  that most implementations ignore, and so does this one -- noted honestly.

FALLBACK CHAINS

`fallbacks=[cheap_provider]` gives you a second provider to try when the primary
is exhausted or broken. This is genuinely valuable -- provider outages are not
rare -- but it is not free: your prompt must work on both models, and quality
will differ. Fall back deliberately, and log loudly when you do.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator, Sequence
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from .cache import LRUCache, is_cacheable, make_cache_key
from .circuit import CircuitBreaker
from .cost import BudgetGuard, CostTracker, estimate_cost
from .errors import CircuitOpenError, LLMError
from .retry import RetryPolicy, with_retry, with_retry_async
from .structured import generate_structured
from .types import LLMProvider, LLMResponse, Message, Usage

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class LLMClient:
    """Production-shaped wrapper around any `LLMProvider`.

    Args:
        provider: primary provider.
        fallbacks: tried in order when the primary fails outright.
        retry: retry policy; `None` disables retry.
        cache: response cache; `None` disables caching.
        budget: pre-flight spend guard; `None` disables it.
        breaker: circuit breaker; `None` disables it.
        default_model: reported when a provider does not name one.
        tenant: cost-attribution key. In a multi-tenant service this comes from
            the request context, not construction.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        fallbacks: Optional[Sequence[LLMProvider]] = None,
        retry: Optional[RetryPolicy] = None,
        cache: Optional[LRUCache] = None,
        budget: Optional[BudgetGuard] = None,
        breaker: Optional[CircuitBreaker] = None,
        tracker: Optional[CostTracker] = None,
        default_model: str = "gpt-4o-mini",
        tenant: str = "default",
    ) -> None:
        self.provider = provider
        self.fallbacks = list(fallbacks or [])
        self.retry_policy = retry if retry is not None else RetryPolicy()
        self.cache = cache
        self.budget = budget
        self.breaker = breaker
        self.tracker = tracker or (budget.tracker if budget else CostTracker())
        self.default_model = default_model
        self.tenant = tenant

    # -- helpers -----------------------------------------------------------

    def _providers(self) -> list[LLMProvider]:
        return [self.provider, *self.fallbacks]

    def _model_of(self, provider: LLMProvider, override: Optional[str]) -> str:
        return override or getattr(provider, "model", None) or self.default_model

    def _finalize(
        self,
        response: LLMResponse,
        *,
        model: str,
        started: float,
        correlation_id: str,
        cache_key: Optional[str],
    ) -> LLMResponse:
        """Attach cost, record it, populate the cache, and log one line."""
        cost = estimate_cost(response.usage, model)
        enriched = LLMResponse(
            text=response.text,
            usage=response.usage,
            model=model,
            finish_reason=response.finish_reason,
            latency_ms=(time.monotonic() - started) * 1000.0,
            cached=False,
            tool_calls=response.tool_calls,
            cost_usd=cost,
        )
        self.tracker.record(response.usage, model, tenant=self.tenant)
        if cache_key is not None and self.cache is not None:
            self.cache.set(cache_key, enriched)

        # One structured line per call. Correlation id is what lets you stitch
        # this to the upstream request (and, across services, to the Java
        # gateway's trace). Never log prompt content at INFO -- it is user data.
        logger.info(
            "llm call cid=%s model=%s in=%d out=%d cost=$%.6f latency=%.0fms finish=%s",
            correlation_id,
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            cost,
            enriched.latency_ms,
            response.finish_reason,
        )
        return enriched

    def _guarded(self, fn: Callable[[], LLMResponse]) -> LLMResponse:
        """Run `fn` through the breaker (if any), then retry inside it."""
        if self.breaker is None:
            return with_retry(fn, policy=self.retry_policy)
        return self.breaker.call(lambda: with_retry(fn, policy=self.retry_policy))

    # -- public API --------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Single completion with the full production pipeline applied."""
        cid = correlation_id or uuid.uuid4().hex[:12]
        started = time.monotonic()
        resolved_model = self._model_of(self.provider, model)

        # Cache first when the request is deterministic: a hit must not be
        # charged against the budget, because it spends nothing.
        cache_key: Optional[str] = None
        if self.cache is not None and is_cacheable(temperature):
            cache_key = make_cache_key(
                messages, resolved_model, temperature=temperature, max_tokens=max_tokens
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                logger.info("llm cache hit cid=%s model=%s", cid, resolved_model)
                return hit

        if self.budget is not None:
            self.budget.check_request(messages, resolved_model)

        last_error: Optional[BaseException] = None
        for index, provider in enumerate(self._providers()):
            provider_model = self._model_of(provider, model)
            call_kwargs = dict(kwargs)
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens

            try:
                response = self._guarded(
                    # Every loop variable is bound as a default argument. Python
                    # closures capture by reference, so without this the lambda
                    # would read whatever `provider_model` holds when it finally
                    # runs -- the classic late-binding bug, and a real one here
                    # because retry defers execution.
                    lambda p=provider, k=call_kwargs, m=provider_model: p.complete(  # type: ignore[misc]
                        messages, model=m, temperature=temperature, **k
                    )
                )
            except (LLMError, CircuitOpenError) as exc:
                last_error = exc
                if index < len(self._providers()) - 1:
                    # Loud on purpose: a silent fallback hides a provider outage
                    # and quietly changes output quality underneath you.
                    logger.warning(
                        "provider %s failed (%s); falling back to %s",
                        getattr(provider, "name", "?"),
                        type(exc).__name__,
                        getattr(self._providers()[index + 1], "name", "?"),
                    )
                    continue
                raise
            else:
                return self._finalize(
                    response,
                    model=provider_model,
                    started=started,
                    correlation_id=cid,
                    cache_key=cache_key,
                )

        raise last_error or LLMError("no provider available")

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion. Same pipeline, non-blocking backoff."""
        cid = correlation_id or uuid.uuid4().hex[:12]
        started = time.monotonic()
        resolved_model = self._model_of(self.provider, model)

        cache_key: Optional[str] = None
        if self.cache is not None and is_cacheable(temperature):
            cache_key = make_cache_key(
                messages, resolved_model, temperature=temperature, max_tokens=max_tokens
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        if self.budget is not None:
            self.budget.check_request(messages, resolved_model)

        last_error: Optional[BaseException] = None
        providers = self._providers()
        for index, provider in enumerate(providers):
            provider_model = self._model_of(provider, model)

            async def attempt(p: LLMProvider = provider, m: str = provider_model) -> LLMResponse:
                return await p.acomplete(messages, model=m, temperature=temperature, **kwargs)

            try:
                if self.breaker is None:
                    response = await with_retry_async(attempt, policy=self.retry_policy)
                else:
                    response = await self.breaker.call_async(
                        lambda: with_retry_async(attempt, policy=self.retry_policy)
                    )
            except (LLMError, CircuitOpenError) as exc:
                last_error = exc
                if index < len(providers) - 1:
                    logger.warning("provider failed (%s); falling back", type(exc).__name__)
                    continue
                raise
            else:
                return self._finalize(
                    response,
                    model=provider_model,
                    started=started,
                    correlation_id=cid,
                    cache_key=cache_key,
                )

        raise last_error or LLMError("no provider available")

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text chunks.

        Deliberately NOT retried. Once the first chunk has reached the client,
        a retry would replay text the user has already seen. Streams fail fast;
        the caller decides whether to restart the whole exchange.
        """
        model_name = self._model_of(self.provider, model)
        if self.budget is not None:
            self.budget.check_request(messages, model_name)
        yield from self.provider.stream(
            messages, model=model_name, temperature=temperature, **kwargs
        )

    def structured(
        self,
        prompt: str,
        schema: type[M],
        *,
        system_prompt: Optional[str] = None,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> M:
        """Validated object out, with bounded self-correction."""
        return generate_structured(
            lambda msgs: self.complete(msgs, **kwargs),
            prompt,
            schema,
            system_prompt=system_prompt,
            max_attempts=max_attempts,
        )

    # -- reporting ---------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return self.tracker.total_usd

    @property
    def total_usage(self) -> Usage:
        return self.tracker.total_usage

    def report(self) -> str:
        lines = [self.tracker.report()]
        if self.cache is not None:
            lines.append(
                f"cache hits={self.cache.hits} misses={self.cache.misses} "
                f"hit_rate={self.cache.hit_rate:.1%}"
            )
        if self.breaker is not None:
            lines.append(
                f"breaker state={self.breaker.state.value} rejected={self.breaker.rejected_calls}"
            )
        return "\n".join(lines)


def build_client(
    provider_name: str = "fake",
    *,
    model: Optional[str] = None,
    daily_budget_usd: Optional[float] = None,
    cache_size: int = 500,
    **provider_kwargs: Any,
) -> LLMClient:
    """Construct a sensibly-configured client from a provider name.

    Real providers are imported lazily so that `build_client("fake")` -- the
    default everywhere in this repo -- never needs an SDK, a key, or a network.
    """
    provider: LLMProvider
    if provider_name == "fake":
        from .providers.fake import FakeProvider

        provider = FakeProvider(**provider_kwargs)  # type: ignore[assignment]
    elif provider_name == "openai":
        from .providers.openai import OpenAIProvider

        provider = OpenAIProvider(model=model or "gpt-4o-mini", **provider_kwargs)
    elif provider_name == "anthropic":
        from .providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(model=model or "claude-3-5-haiku-20241022", **provider_kwargs)
    else:
        raise ValueError(f"unknown provider {provider_name!r}; expected fake|openai|anthropic")

    return LLMClient(
        provider,
        cache=LRUCache(max_size=cache_size),
        budget=BudgetGuard(daily_budget_usd) if daily_budget_usd else None,
        breaker=CircuitBreaker(),
        default_model=model or "gpt-4o-mini",
    )
