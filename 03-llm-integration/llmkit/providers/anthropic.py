"""Anthropic (Claude) provider on raw httpx.

Deliberately structured identically to `openai.py` so the genuine wire
differences are visible rather than buried in two different SDKs. There are
exactly five that matter:

| Concern        | OpenAI                          | Anthropic                        |
|----------------|---------------------------------|----------------------------------|
| Endpoint       | POST /v1/chat/completions       | POST /v1/messages                |
| Auth header    | Authorization: Bearer sk-...    | x-api-key: sk-ant-...            |
| Versioning     | in the URL path                 | anthropic-version header (req'd) |
| System prompt  | first element of messages[]     | TOP-LEVEL `system` field         |
| max_tokens     | optional                        | REQUIRED                         |

The system-prompt difference is the one that bites during a migration: pass a
system message inside `messages` to Anthropic and you get a 400, because only
"user" and "assistant" are legal roles there. `split_system()` in types.py
exists for exactly this.

The streaming format also differs: Anthropic sends named SSE event types
(`content_block_delta`, `message_delta`, `message_stop`) rather than OpenAI's
single anonymous chunk shape with a `[DONE]` sentinel.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any, Optional

import httpx

from ..errors import LLMError, TransientError, classify_status
from ..types import LLMResponse, Message, Usage, split_system

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_TIMEOUT = 30.0
API_VERSION = "2023-06-01"

# Anthropic requires max_tokens. Picking a default rather than erroring keeps the
# provider drop-in compatible with the OpenAI one, which does not.
DEFAULT_MAX_TOKENS = 4096


def _retry_after(response: httpx.Response) -> Optional[float]:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class AnthropicProvider:
    """Messages API over raw HTTP."""

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client = client
        self._async_client = async_client

    # -- plumbing ----------------------------------------------------------

    @property
    def api_key(self) -> str:
        key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set (and no api_key was passed)")
        return key

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            # Omitting this header is a 400. It is the most common first-time
            # mistake with this API.
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str],
        temperature: float,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # The migration trap: system must be hoisted out of messages[].
        system_text, rest = split_system(messages)
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in rest],
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if system_text:
            payload["system"] = system_text
        if stream:
            payload["stream"] = True
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
            request_id=response.headers.get("request-id"),
        )

    @staticmethod
    def _parse(data: dict[str, Any], started: float) -> LLMResponse:
        # Content is a LIST of typed blocks, not a string -- Anthropic is
        # multimodal-shaped by default. Concatenate the text blocks.
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        stop_reason = data.get("stop_reason") or "stop"
        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            model=data.get("model", ""),
            # Normalize Anthropic's vocabulary onto ours:
            # end_turn->stop, max_tokens->length, tool_use->tool_use.
            finish_reason={
                "end_turn": "stop",
                "stop_sequence": "stop",
                "max_tokens": "length",
                "tool_use": "tool_use",
            }.get(stop_reason, stop_reason),
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
                f"{self.base_url}/messages",
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
                f"{self.base_url}/messages",
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
        """Yield text deltas from Anthropic's named-event SSE stream.

            event: content_block_delta
            data: {"delta":{"type":"text_delta","text":"Hel"}}

            event: message_stop
            data: {}

        Unlike OpenAI there is no `[DONE]` sentinel -- the stream ends on
        `message_stop`, or simply when the body closes.
        """
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            with client.stream(
                "POST",
                f"{self.base_url}/messages",
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
                        continue  # skip `event:` lines; the payload is self-describing
                    try:
                        chunk = json.loads(line[len("data:") :].strip())
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("type") == "message_stop":
                        break
                    delta = chunk.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
        except httpx.TimeoutException as exc:
            raise TransientError("stream timed out", provider=self.name) from exc
        finally:
            if self._client is None:
                client.close()
