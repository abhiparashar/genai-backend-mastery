"""llmkit -- a provider-agnostic LLM client with production behaviour built in.

    from llmkit import build_client, user

    client = build_client("fake")              # swap to "openai" / "anthropic"
    print(client.complete([user("Hello")]).text)

What you get over calling a vendor SDK directly:

- retry with full jitter, honouring Retry-After, on ONLY the errors worth retrying
- a circuit breaker so a provider outage fails fast instead of storming
- a fallback provider chain
- response caching that refuses to cache non-deterministic requests
- per-request and per-tenant cost accounting, plus a pre-flight budget ceiling
- typed, validated structured output with bounded self-correction
- one structured log line per call, carrying a correlation id

Only the offline `FakeProvider` is imported eagerly. `openai` and `anthropic`
are resolved lazily inside `build_client`, so importing this package never
requires a network, an SDK, or an API key.
"""

from __future__ import annotations

from .cache import LRUCache, is_cacheable, make_cache_key
from .circuit import CircuitBreaker, CircuitOpenError, State
from .client import LLMClient, build_client
from .cost import PRICES, BudgetGuard, CostTracker, count_tokens, estimate_cost
from .errors import (
    AuthError,
    BudgetExceededError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    RateLimitError,
    TransientError,
    classify_status,
)
from .providers.fake import FakeProvider
from .retry import RetryPolicy, compute_delay, with_retry, with_retry_async
from .structured import (
    StructuredOutputError,
    extract_json,
    generate_structured,
    repair_json,
)
from .types import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
    assistant,
    system,
    user,
)

__version__ = "1.0.0"

__all__ = [
    # types
    "Message",
    "Usage",
    "ToolCall",
    "LLMResponse",
    "LLMProvider",
    "system",
    "user",
    "assistant",
    # client
    "LLMClient",
    "build_client",
    "FakeProvider",
    # errors
    "LLMError",
    "RateLimitError",
    "TransientError",
    "AuthError",
    "InvalidRequestError",
    "ContextLengthError",
    "ContentFilterError",
    "BudgetExceededError",
    "CircuitOpenError",
    "classify_status",
    # resilience
    "RetryPolicy",
    "with_retry",
    "with_retry_async",
    "compute_delay",
    "CircuitBreaker",
    "State",
    # cost
    "CostTracker",
    "BudgetGuard",
    "estimate_cost",
    "count_tokens",
    "PRICES",
    # cache
    "LRUCache",
    "make_cache_key",
    "is_cacheable",
    # structured output
    "generate_structured",
    "StructuredOutputError",
    "extract_json",
    "repair_json",
    "__version__",
]
