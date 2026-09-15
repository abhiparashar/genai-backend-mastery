"""A deterministic, offline LLM provider.

WHY THIS IS THE MOST IMPORTANT FILE IN THE TEST SUITE

Unit tests must never call a real LLM. Three reasons, in order of how much pain
they cause:

1. **Non-determinism.** A real model returns different text every run. You cannot
   assert on it, so people write `assert len(response) > 0`, which passes even
   when the system is catastrophically broken.
2. **Cost and latency.** A 400-test suite at 2s and $0.002 per call is 13 minutes
   and $0.80 *per run*. Nobody runs that on every commit, so it rots.
3. **Failure coverage.** You cannot ask OpenAI to return a 429 on demand. But
   rate-limit handling is exactly the code most likely to be wrong, so it is
   exactly the code you must be able to trigger at will.

`FakeProvider` fixes all three: deterministic output, instant, free, and you can
script any failure sequence you like.

The Java analogue is Mockito — with the difference that this is a real
implementation of the interface, not a proxy, so it also doubles as runnable
documentation and powers every example in this repo offline.

    provider = FakeProvider(responses=["Paris"])
    provider.complete([user("Capital of France?")]).text   # -> "Paris"

    # Fail twice with 429, then succeed: proves your retry logic works.
    provider = FakeProvider(fail_times=2, failure=RateLimitError("429", retry_after=0))
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from ..errors import LLMError
from ..types import LLMResponse, Message, Usage

# Rough token estimate: ~4 characters per token for English prose. Real
# tokenizers (tiktoken/BPE) differ, but this is stable, dependency-free, and
# within ~10% for ordinary text -- good enough for tests and for cost estimates
# when tiktoken is not installed. See llmkit/cost.py for the real thing.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Cheap token estimate. Always at least 1 for non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


class FakeProvider:
    """Scriptable in-memory provider.

    Args:
        responses: Replies returned in order. When exhausted, it falls back to a
            deterministic echo derived from the prompt hash, so a test that makes
            an unexpected extra call still gets stable output instead of an
            IndexError three layers down.
        fail_times: Raise `failure` for the first N calls, then behave normally.
            This is how you test retry, fallback, and circuit-breaker logic.
        failure: The exception to raise while failing. Defaults to a retryable
            TransientError.
        latency: Simulated seconds per call. Keep at 0.0 in unit tests; set it in
            concurrency demos so `asyncio.gather` visibly beats sequential calls.
        model: Reported model name.
    """

    name = "fake"

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        *,
        fail_times: int = 0,
        failure: Optional[BaseException] = None,
        latency: float = 0.0,
        model: str = "fake-1",
    ) -> None:
        self._responses = list(responses or [])
        # Public and mutable on purpose: demos flip this to 0 to simulate a
        # provider recovering, which is how the circuit-breaker example shows
        # HALF_OPEN closing again.
        self.fail_times = fail_times
        self._failure = failure or LLMError("simulated upstream failure")
        # A retryable default: most tests want to exercise the retry path.
        if failure is None:
            self._failure.retryable = True  # type: ignore[attr-defined]
        self.latency = latency
        self.model = model

        # Observability for assertions: how many calls, and with what.
        self.calls = 0
        self.failures = 0
        self.succeeded = 0  # drives scripted-response indexing
        self.prompts: list[list[Message]] = []

    # -- introspection helpers used by tests ------------------------------

    @property
    def last_prompt(self) -> Optional[list[Message]]:
        return self.prompts[-1] if self.prompts else None

    def reset(self) -> None:
        self.calls = 0
        self.failures = 0
        self.succeeded = 0
        self.prompts.clear()

    # -- internals --------------------------------------------------------

    def _next_text(self, messages: Sequence[Message]) -> str:
        """Pick the scripted reply, else a deterministic synthetic one.

        Indexed by SUCCESSFUL calls, not total calls. A simulated failure must
        not burn a scripted response -- otherwise `fail_times=2` combined with
        `responses=["ok"]` would retry into the fallback text, and the test
        would silently assert on the wrong thing.
        """
        index = self.succeeded
        self.succeeded += 1
        if index < len(self._responses):
            return self._responses[index]

        # Deterministic fallback: same prompt always yields the same text, which
        # keeps cache tests meaningful (a cache hit must be indistinguishable).
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        digest = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
        return f"fake response to {last_user[:40]!r} [{digest}]"

    def _maybe_fail(self) -> None:
        if self.failures < self.fail_times:
            self.failures += 1
            raise self._failure

    def _build(self, messages: Sequence[Message], text: str, started: float) -> LLMResponse:
        prompt_chars = sum(len(m.content) for m in messages)
        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=estimate_tokens("x" * prompt_chars),
                output_tokens=estimate_tokens(text),
            ),
            model=self.model,
            finish_reason="stop",
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    # -- LLMProvider interface --------------------------------------------

    def complete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        started = time.monotonic()
        self.calls += 1
        self.prompts.append(list(messages))
        self._maybe_fail()
        if self.latency:
            time.sleep(self.latency)
        return self._build(messages, self._next_text(messages), started)

    async def acomplete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        started = time.monotonic()
        self.calls += 1
        self.prompts.append(list(messages))
        self._maybe_fail()
        if self.latency:
            # asyncio.sleep, never time.sleep: blocking here would serialize
            # every concurrent request in the process and silently defeat the
            # whole point of the async demos.
            await asyncio.sleep(self.latency)
        return self._build(messages, self._next_text(messages), started)

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterator[str]:
        """Yield the reply word by word, mimicking real token streaming.

        Chunk boundaries from a real provider are sub-word tokens, not words, so
        never assume a chunk is a complete word in consuming code. Streaming
        deltas reassemble by plain concatenation -- that is the only contract.
        """
        self.calls += 1
        self.prompts.append(list(messages))
        self._maybe_fail()
        text = self._next_text(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            if self.latency:
                time.sleep(self.latency / max(1, len(words)))
            yield word if i == len(words) - 1 else word + " "
