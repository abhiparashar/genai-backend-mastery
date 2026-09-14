"""Core value types shared by every provider.

WHY these exist: the vendor SDKs each invent their own response object. If you let
`openai.types.chat.ChatCompletion` leak into your business logic, swapping to Claude
becomes a refactor instead of a config change. These types are the anti-corruption
layer -- the same reason you'd map a JPA entity to a domain object in Java.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

# Roles are the wire vocabulary shared by OpenAI and Anthropic.
ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class Message:
    """One turn in a conversation.

    Frozen because a message that mutates after it entered a conversation history
    is a debugging nightmare -- the log says one thing, the replay another.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"invalid role {self.role!r}; expected one of {ROLES}")

    def to_wire(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


@dataclass(frozen=True)
class Usage:
    """Token accounting. The unit of cost in every LLM system."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool. `arguments` is already JSON-parsed."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized completion, identical in shape across providers."""

    text: str
    usage: Usage
    model: str
    finish_reason: str = "stop"  # stop | length | tool_use | error
    latency_ms: float = 0.0
    cached: bool = False
    tool_calls: Sequence[ToolCall] = ()
    cost_usd: float = 0.0

    @property
    def truncated(self) -> bool:
        """True when the model hit max_tokens mid-thought.

        Worth checking explicitly: a truncated JSON response is the single most
        common cause of "the parser randomly fails in production".
        """
        return self.finish_reason == "length"


class LLMProvider:
    """Structural interface every provider satisfies.

    Deliberately a plain base class rather than `typing.Protocol` so that
    `isinstance` works at runtime on Python 3.9 and subclasses get a clear
    contract violation (NotImplementedError) instead of an AttributeError
    three frames deep.
    """

    name: str = "base"

    def complete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def acomplete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError


def messages_to_wire(messages: Sequence[Message]) -> List[Dict[str, str]]:
    return [m.to_wire() for m in messages]


def split_system(messages: Sequence[Message]) -> "tuple[Optional[str], List[Message]]":
    """Split a leading system message out of the list.

    Needed because Anthropic takes `system` as a TOP-LEVEL request field, while
    OpenAI takes it as the first element of `messages`. A small wire difference
    that bites everyone porting between the two.
    """
    system_text: Optional[str] = None
    rest: List[Message] = []
    for m in messages:
        if m.role == "system" and system_text is None and not rest:
            system_text = m.content
        else:
            rest.append(m)
    return system_text, rest
