"""Request and response models.

Pydantic is Bean Validation + Lombok + Jackson in one object: it declares the
shape, validates it, generates the OpenAPI schema, and serialises the result.
FastAPI rejects a bad body with a 422 and a precise field-level error before
your handler is ever entered -- the `@Valid @RequestBody` equivalent.

VALIDATE AT THE EDGE, ALWAYS

Every constraint here exists to stop something expensive:

    max prompt length   an unbounded prompt is an unbounded bill, and the
                        cheapest denial-of-wallet attack there is
    model allowlist     otherwise a caller selects gpt-4o on your budget
    max_tokens ceiling  bounds the output side, which costs 4x the input
    temperature bounds  values outside 0-2 are a provider 400 you can avoid

NOTE ON 3.9: Pydantic evaluates annotations at RUNTIME even with
`from __future__ import annotations`, so `Optional[X]` is required here --
`X | None` raises on 3.9. See PROGRESS.md.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Allowlist, not free text. A caller must not be able to name an arbitrary
# model and bill you for it.
ALLOWED_MODELS = (
    "fake-1",
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
)


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatRequest(BaseModel):
    """Body for POST /v1/chat and /v1/chat/stream.

    `messages[]` rather than a bare prompt: it maps 1:1 onto both provider
    payloads, is multi-turn native, and extends to tool roles without a
    breaking change. A `prompt` + `system` shape cannot express an assistant
    turn, so history has to be smuggled in out of band.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Summarise our refund policy."}],
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                }
            ]
        }
    }

    messages: list[MessageIn] = Field(min_length=1, max_length=100)
    model: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=8192)
    conversation_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("model")
    @classmethod
    def _model_must_be_allowlisted(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ALLOWED_MODELS:
            raise ValueError(f"model must be one of: {', '.join(ALLOWED_MODELS)}")
        return value

    @field_validator("messages")
    @classmethod
    def _must_end_with_a_user_turn(cls, value: list) -> list:
        # A conversation ending on an assistant message means the caller
        # forgot to append the new question; the model would answer itself.
        if value and value[-1].role == "assistant":
            raise ValueError("the final message must be from the user or system")
        return value

    def total_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    text: str
    model: str
    finish_reason: str = "stop"
    usage: UsageOut
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    correlation_id: str = "-"


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    result: Optional[ChatResponse] = None
    error: Optional[dict] = None


class ErrorDetail(BaseModel):
    code: Literal[
        "rate_limited",
        "invalid_request",
        "unauthorized",
        "budget_exceeded",
        "upstream_error",
        "payload_too_large",
        "internal",
    ]
    message: str
    correlation_id: str = "-"


class ErrorResponse(BaseModel):
    """One error envelope for every non-2xx.

    A single documented shape means clients write one error handler instead of
    guessing per endpoint. The message is always safe to show a user: internal
    detail goes to the logs, keyed by correlation_id.
    """

    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, Any]
