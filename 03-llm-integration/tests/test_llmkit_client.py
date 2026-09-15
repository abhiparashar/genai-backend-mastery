"""Caching, cost accounting, budget enforcement, structured output, and the
client façade that composes them.

Every test here asserts something a consumer can observe: what got charged,
what reached the provider, what the caller received back.
"""

from __future__ import annotations

import pytest
from llmkit.cache import LRUCache, is_cacheable, make_cache_key
from llmkit.circuit import CircuitBreaker, manual_clock
from llmkit.client import LLMClient
from llmkit.cost import BudgetGuard, CostTracker, count_message_tokens, estimate_cost
from llmkit.errors import BudgetExceededError, CircuitOpenError, TransientError
from llmkit.providers import FakeProvider
from llmkit.retry import RetryPolicy
from llmkit.structured import (
    StructuredOutputError,
    extract_json,
    generate_structured,
    repair_json,
)
from llmkit.types import LLMResponse, Message, Usage, user
from pydantic import BaseModel, Field

FAST = RetryPolicy(max_attempts=3, base_delay=0.001, jitter=False)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_key_covers_every_output_affecting_parameter():
    base = make_cache_key([user("hi")], "gpt-4o-mini")
    # Omitting any of these from the key is a cache-poisoning bug: a request
    # would be served an answer generated under different parameters.
    assert base != make_cache_key([user("hi")], "gpt-4o")
    assert base != make_cache_key([user("hi")], "gpt-4o-mini", temperature=0.7)
    assert base != make_cache_key([user("hi")], "gpt-4o-mini", max_tokens=50)
    assert base != make_cache_key([user("bye")], "gpt-4o-mini")
    assert base == make_cache_key([user("hi")], "gpt-4o-mini"), "key must be stable"


def test_only_deterministic_requests_are_cacheable():
    assert is_cacheable(0.0) is True
    assert is_cacheable(0.7) is False


def test_cache_hit_reports_zero_usage_so_spend_is_not_double_counted():
    cache = LRUCache()
    cache.set("k", LLMResponse("answer", Usage(100, 50), "gpt-4o-mini"))
    hit = cache.get("k")
    assert hit is not None
    assert hit.text == "answer"
    assert hit.cached is True
    assert hit.usage.total_tokens == 0, "a cache hit called no provider, so it billed nothing"


def test_cache_evicts_least_recently_used():
    cache = LRUCache(max_size=2)
    response = LLMResponse("x", Usage(1, 1), "m")
    cache.set("a", response)
    cache.set("b", response)
    cache.get("a")  # 'a' is now the most recently used, so 'b' should go
    cache.set("c", response)
    assert cache.get("b") is None
    assert cache.get("a") is not None


def test_cache_entries_expire():
    cache = LRUCache(ttl=10)
    cache.set("k", LLMResponse("x", Usage(1, 1), "m"), now=0.0)
    assert cache.get("k", now=9.0) is not None
    assert cache.get("k", now=11.0) is None


def test_client_serves_repeat_requests_from_cache():
    client = LLMClient(FakeProvider(), cache=LRUCache(), default_model="gpt-4o-mini")
    first = client.complete([user("same question")])
    second = client.complete([user("same question")])

    assert second.text == first.text
    assert second.cached is True
    assert client.provider.calls == 1, "the second call must not reach the provider"


def test_client_bypasses_cache_when_sampling_is_random():
    client = LLMClient(FakeProvider(), cache=LRUCache(), default_model="gpt-4o-mini")
    client.complete([user("q")], temperature=0.9)
    client.complete([user("q")], temperature=0.9)
    assert client.provider.calls == 2, "temperature>0 means the caller WANTS variation"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_is_priced_separately_for_input_and_output():
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out. Output costs 4x input, which is
    # why "be concise" in a system prompt is a real cost lever.
    assert estimate_cost(Usage(1_000_000, 0), "gpt-4o-mini") == pytest.approx(0.15)
    assert estimate_cost(Usage(0, 1_000_000), "gpt-4o-mini") == pytest.approx(0.60)


def test_unknown_model_costs_zero_rather_than_guessing():
    assert estimate_cost(Usage(1000, 1000), "some-new-model") == 0.0


def test_message_token_count_includes_chat_format_overhead():
    # Raw content tokens alone under-count; the API adds role/delimiter tokens.
    messages = [Message("user", "hello")]
    assert count_message_tokens(messages) > 1


def test_tracker_attributes_spend_by_tenant():
    tracker = CostTracker()
    tracker.record(Usage(1_000_000, 0), "gpt-4o-mini", tenant="acme")
    tracker.record(Usage(1_000_000, 0), "gpt-4o-mini", tenant="globex")
    tracker.record(Usage(1_000_000, 0), "gpt-4o-mini", tenant="acme")

    by_tenant = tracker.by_tenant()
    assert by_tenant["acme"] == pytest.approx(0.30)
    assert by_tenant["globex"] == pytest.approx(0.15)
    assert tracker.total_usd == pytest.approx(0.45)


def test_budget_guard_blocks_before_the_call_is_made():
    client = LLMClient(FakeProvider(), budget=BudgetGuard(0.000001))
    with pytest.raises(BudgetExceededError):
        client.complete([user("x" * 5000)], model="gpt-4o")
    assert client.provider.calls == 0, "a guard that fires after the spend is an invoice"


def test_budget_guard_allows_calls_within_the_cap():
    client = LLMClient(FakeProvider(responses=["ok"]), budget=BudgetGuard(10.0))
    assert client.complete([user("hi")], model="gpt-4o-mini").text == "ok"


# ---------------------------------------------------------------------------
# Client composition
# ---------------------------------------------------------------------------


def test_response_carries_cost_and_latency():
    client = LLMClient(FakeProvider(responses=["hi"]))
    response = client.complete([user("q")], model="gpt-4o-mini")
    assert response.cost_usd >= 0.0
    assert response.latency_ms >= 0.0
    assert response.model == "gpt-4o-mini"


def test_falls_back_to_the_secondary_provider():
    primary = FakeProvider(fail_times=99, failure=TransientError("503"))
    secondary = FakeProvider(responses=["from backup"])
    client = LLMClient(primary, fallbacks=[secondary], retry=FAST)

    assert client.complete([user("hi")]).text == "from backup"
    assert secondary.calls == 1


def test_fallback_is_not_used_when_the_primary_succeeds():
    primary = FakeProvider(responses=["primary"])
    secondary = FakeProvider(responses=["secondary"])
    client = LLMClient(primary, fallbacks=[secondary], retry=FAST)

    assert client.complete([user("hi")]).text == "primary"
    assert secondary.calls == 0


def test_open_breaker_short_circuits_the_provider():
    provider = FakeProvider(fail_times=99, failure=TransientError("503"))
    client = LLMClient(
        provider,
        retry=RetryPolicy(max_attempts=1),
        breaker=CircuitBreaker(failure_threshold=2, clock=manual_clock()),
    )
    for _ in range(2):
        with pytest.raises(TransientError):
            client.complete([user("x")])

    calls_before = provider.calls
    with pytest.raises(CircuitOpenError):
        client.complete([user("x")])
    assert provider.calls == calls_before, "an open circuit must not reach the provider"


def test_streaming_chunks_reassemble_into_the_full_text():
    client = LLMClient(FakeProvider(responses=["one two three"]))
    chunks = list(client.stream([user("go")]))
    assert len(chunks) > 1, "streaming must actually be incremental"
    assert "".join(chunks) == "one two three"


@pytest.mark.asyncio
async def test_async_completion_shares_the_same_pipeline():
    client = LLMClient(FakeProvider(responses=["async ok"]), cache=LRUCache())
    first = await client.acomplete([user("q")])
    second = await client.acomplete([user("q")])
    assert first.text == "async ok"
    assert second.cached is True


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    vendor: str
    total: float
    line_items: int = Field(ge=0)


@pytest.mark.parametrize(
    "raw",
    [
        '{"vendor":"Acme","total":42.5,"line_items":3}',
        'Sure!\n```json\n{"vendor":"Acme","total":42.5,"line_items":3}\n```\nAnything else?',
        'Here: {"vendor":"Acme","total":42.5,"line_items":3,}',
    ],
    ids=["clean", "fenced-with-prose", "trailing-comma"],
)
def test_structured_output_survives_the_ways_models_actually_reply(raw):
    provider = FakeProvider(responses=[raw])
    invoice = generate_structured(provider.complete, "extract", Invoice)
    assert invoice.vendor == "Acme"
    assert invoice.total == 42.5


def test_extract_json_is_string_aware():
    # A brace inside a string value must not terminate the object early.
    assert extract_json('prefix {"k":"a}b"} suffix') == '{"k":"a}b"}'
    assert extract_json("no json here") is None


def test_repair_converts_python_literals():
    assert repair_json('{"a": 1,}') == '{"a": 1}'
    assert "true" in repair_json('{"a": True}')


def test_invalid_output_triggers_a_bounded_re_ask():
    provider = FakeProvider(
        responses=[
            '{"vendor":"Acme","total":"not a number","line_items":-1}',
            '{"vendor":"Acme","total":7.0,"line_items":2}',
        ]
    )
    invoice = generate_structured(provider.complete, "extract", Invoice, max_attempts=3)
    assert invoice.total == 7.0
    assert provider.calls == 2, "it should stop as soon as validation passes"


def test_gives_up_after_the_attempt_budget():
    provider = FakeProvider(responses=["garbage", "still garbage", "nope"])
    with pytest.raises(StructuredOutputError) as exc:
        generate_structured(provider.complete, "extract", Invoice, max_attempts=3)
    assert exc.value.attempts == 3


def test_truncated_response_fails_immediately_with_the_real_cause():
    # Re-asking cannot help: without raising max_tokens it will truncate again.
    class Truncating:
        def complete(self, messages, **kwargs):
            return LLMResponse('{"vendor":"Acme"', Usage(1, 1), "m", finish_reason="length")

    with pytest.raises(StructuredOutputError, match="truncated"):
        generate_structured(Truncating().complete, "extract", Invoice, max_attempts=3)


def test_structured_through_the_client_facade():
    client = LLMClient(FakeProvider(responses=['{"vendor":"Beta","total":1.0,"line_items":0}']))
    assert client.structured("extract", Invoice).vendor == "Beta"
