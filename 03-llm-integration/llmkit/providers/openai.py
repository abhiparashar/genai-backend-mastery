"""OpenAI provider built directly on httpx -- no vendor SDK.

WHY NOT JUST USE THE `openai` PACKAGE?

In production you usually should. The reason this repo implements the wire
protocol by hand is pedagogical and it is the whole point of the module:

- You see that a "chat completion" is one POST with a JSON body. The mystique
  disappears, and so does the fear of debugging it.
- You can read the SSE streaming format instead of treating it as magic.
- You learn where the retryable signals live (status codes, `Retry-After`,
  `x-request-id`) -- the SDK hides these behind its own exception types, and
  when you need to tune retry behaviour you need to know them anyway.
- Zero dependency: this file imports nothing the repo does not already need.

The Anthropic provider next door is deliberately near-identical, which makes
the handful of genuine wire differences (top-level `system`, `max_tokens`
required, different SSE event names) stand out instead of hiding inside two
different SDKs.

AUTH: `Authorization: Bearer sk-...`, read from OPENAI_API_KEY at call time --
never at import time, so importing this module never fails on a missing key.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any, Optional

import httpx

from ..errors import LLMError, TransientError, classify_status
from ..types import LLMResponse, Message, Usage, messages_to_wire

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 30.0


def _retry_after(response: httpx.Response) -> Optional[float]:
    """Parse Retry-After. Seconds form only; HTTP-date form is rare here."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class OpenAIProvider:
    """Chat Completions over raw HTTP.

    Args:
        model: default model for calls that do not override it.
        api_key: falls back to $OPENAI_API_KEY, resolved lazily.
        base_url: override to point at a compatible server. This is why
            OpenAI-compatible APIs matter -- vLLM, Ollama, Groq, Together and
            most self-hosted servers speak this exact protocol, so the same
            class drives them with one URL change.
        timeout: per-request seconds. Always set one. The default httpx timeout
            is generous and a hung LLM request holds a worker indefinitely.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Reusing a client reuses the TCP/TLS connection pool. Creating one per
        # request costs a full handshake every time -- measurable when you are
        # making thousands of calls.
        self._client = client
        self._async_client = async_client

    # -- plumbing ----------------------------------------------------------

    @property
    def api_key(self) -> str:
        key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set (and no api_key was passed)")
        return key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str],
        temperature: float,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages_to_wire(messages),
            "temperature": temperature,
        }
        if stream:
            payload["stream"] = True
            # Ask for usage on the final SSE chunk; without this, streamed
            # responses report no token counts and your cost tracking silently
            # under-reports every streaming request.
            payload["stream_options"] = {"include_usage": True}
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        raise classify_status(
            response.status_code,
            response.text,
            provider=self.name,
            retry_after=_retry_after(response),
            request_id=response.headers.get("x-request-id"),
        )

    @staticmethod
    def _parse(data: dict[str, Any], started: float) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(choice.get("message") or {}).get("content") or "",
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            model=data.get("model", ""),
            # OpenAI says "stop" | "length" | "tool_calls" | "content_filter".
            # Normalized to our vocabulary so callers never branch on provider.
            finish_reason={"tool_calls": "tool_use"}.get(
                choice.get("finish_reason", "stop"), choice.get("finish_reason", "stop")
            ),
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    # -- LLMProvider -------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        started = time.monotonic()
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, model=model, temperature=temperature, **kwargs),
            )
        except httpx.TimeoutException as exc:
            # A timeout is retryable; an SDK would classify this for you.
            raise TransientError(
                f"request timed out after {self.timeout}s", provider=self.name
            ) from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"connection error: {exc}", provider=self.name) from exc
        finally:
            if self._client is None:
                client.close()

        self._raise_for_status(response)
        return self._parse(response.json(), started)

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        started = time.monotonic()
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, model=model, temperature=temperature, **kwargs),
            )
        except httpx.TimeoutException as exc:
            raise TransientError(
                f"request timed out after {self.timeout}s", provider=self.name
            ) from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"connection error: {exc}", provider=self.name) from exc
        finally:
            if self._async_client is None:
                await client.aclose()

        self._raise_for_status(response)
        return self._parse(response.json(), started)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield content deltas from the SSE stream.

        The wire format is Server-Sent Events, one JSON object per `data:` line:

            data: {"choices":[{"delta":{"content":"Hel"}}]}
            data: {"choices":[{"delta":{"content":"lo"}}]}
            data: [DONE]

        Two traps worth knowing:
        - `[DONE]` is a literal string, NOT JSON. Parsing it blindly throws.
        - Deltas are sub-word tokens. Never assume a chunk is a whole word;
          the only contract is that concatenation reproduces the text.
        """
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(
                    messages, model=model, temperature=temperature, stream=True, **kwargs
                ),
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue  # usage-only final chunk
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise TransientError("stream timed out", provider=self.name) from exc
        finally:
            if self._client is None:
                client.close()

    def embed(
        self, texts: Sequence[str], *, model: str = "text-embedding-3-small"
    ) -> list[list[float]]:
        """Embeddings share the same auth and error handling, different path."""
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json={"model": model, "input": list(texts)},
            )
        finally:
            if self._client is None:
                client.close()
        self._raise_for_status(response)
        return [item["embedding"] for item in response.json()["data"]]
