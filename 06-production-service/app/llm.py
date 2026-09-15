"""The service's LLM layer: contract, fake provider, and cost accounting.

Self-contained (see the standalone-module note in `ragkit/types.py`), and
deliberately minimal -- module 03's `llmkit` is the full-featured version.
What matters here is that the SERVICE boots and serves with no credentials,
so every test and every reader can run it.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

# USD per 1M tokens. Stale by the time you read this -- load from config in
# production so a price change is not a redeploy. See llmkit/cost.py.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "fake-1": (0.0, 0.0),
}

CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = "fake-1"
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cached: bool = False


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def estimate_cost(usage: Usage, model: str) -> float:
    if model not in PRICES:
        return 0.0
    input_price, output_price = PRICES[model]
    return (usage.input_tokens / 1_000_000) * input_price + (
        usage.output_tokens / 1_000_000
    ) * output_price


class LLMError(Exception):
    """Upstream provider failure."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class FakeProvider:
    """Deterministic offline provider.

    Streams word by word so SSE framing can be tested end to end without a
    network, and `fail_times` lets tests drive the 502 path deliberately.
    """

    name = "fake"

    def __init__(
        self,
        model: str = "fake-1",
        *,
        responses: Optional[Sequence[str]] = None,
        fail_times: int = 0,
        latency: float = 0.0,
    ) -> None:
        self.model = model
        self._responses = list(responses or [])
        self.fail_times = fail_times
        self.latency = latency
        self.calls = 0
        self.failures = 0

    def _text(self, messages: Sequence[Message]) -> str:
        index = self.calls - 1
        if index < len(self._responses):
            return self._responses[index]
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        digest = hashlib.sha256(last.encode()).hexdigest()[:8]
        return f"This is a deterministic reply to {last[:60]!r} ({digest})."

    def _maybe_fail(self) -> None:
        if self.failures < self.fail_times:
            self.failures += 1
            raise LLMError("simulated upstream failure", retryable=True)

    async def acomplete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        started = time.monotonic()
        self.calls += 1
        self._maybe_fail()
        if self.latency:
            await asyncio.sleep(self.latency)
        text = self._text(messages)
        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=estimate_tokens(" ".join(m.content for m in messages)),
                output_tokens=estimate_tokens(text),
            ),
            model=self.model,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    async def astream(self, messages: Sequence[Message], **kwargs: Any) -> AsyncIterator[str]:
        self.calls += 1
        self._maybe_fail()
        text = self._text(messages)
        words = text.split(" ")
        for index, word in enumerate(words):
            if self.latency:
                await asyncio.sleep(self.latency / max(1, len(words)))
            yield word if index == len(words) - 1 else word + " "


class OpenAIProvider:
    """Real provider over httpx. Imported lazily; see llmkit for the full one."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _payload(self, messages: Sequence[Message], **kwargs: Any) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return payload

    async def acomplete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        import httpx

        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._payload(messages, **kwargs),
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"connection error: {exc}", retryable=True) from exc

        if response.status_code >= 400:
            # 4xx is our bug and must not be retried; 5xx and 429 may be.
            raise LLMError(
                f"provider returned {response.status_code}",
                retryable=response.status_code >= 500 or response.status_code == 429,
            )

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(choice.get("message") or {}).get("content", ""),
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    async def astream(self, messages: Sequence[Message], **kwargs: Any) -> AsyncIterator[str]:
        import json

        import httpx

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._payload(messages, stream=True, **kwargs),
            ) as response,
        ):
            if response.status_code >= 400:
                await response.aread()
                raise LLMError(
                    f"provider returned {response.status_code}",
                    retryable=response.status_code >= 500,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":  # a literal, not JSON
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta


def build_provider(settings: Any) -> Any:
    """Select a provider from settings, falling back to fake.

    Falls back loudly rather than crashing: a missing key in dev should degrade
    to the offline provider, not prevent the service from starting.
    """
    if settings.llm_provider == "openai" and settings.provider_key():
        return OpenAIProvider(
            settings.provider_key(), settings.llm_model, settings.llm_timeout_seconds
        )
    return FakeProvider(model="fake-1" if settings.llm_provider == "fake" else settings.llm_model)
