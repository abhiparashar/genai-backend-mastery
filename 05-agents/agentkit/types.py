"""Core types for agents, plus a scriptable offline provider.

WHAT AN AGENT IS (AND WHY THE DEFINITION MATTERS)

A chain is a fixed pipeline: retrieve, then summarise, then format. You wrote
the order.

An agent is a **loop where the model chooses the next step**:

    think -> act (call a tool) -> observe the result -> repeat until done

That single difference -- the model controls control flow -- is where all the
power and all the danger comes from. A chain has a known cost and a known
failure set. An agent can loop, spend unboundedly, call the wrong tool with
the wrong arguments, or be talked into something by text it retrieved.

WHEN NOT TO USE AN AGENT

Most of the time. Seriously.

A 10-step agent run costs roughly 10x a single call, takes 10x the latency,
and has 10x the surfaces on which to fail. If you can express the task as a
fixed sequence, write the chain -- it is cheaper, faster, testable, and
debuggable.

Use an agent only when the number or order of steps genuinely cannot be known
in advance. "Look up this order, and if it shipped late, check the carrier's
status, and if that failed, draft an apology" is an agent. "Summarise this
document" is not.

Every type here exists to make an agent run OBSERVABLE: `Step` records the
reasoning, `AgentResult` records the cost, and `stop_reason` records why it
ended -- because "it finished" and "it hit the iteration cap" look identical
from the outside, and only one of them is success.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

ROLES = ("system", "user", "assistant", "tool")


# ---------------------------------------------------------------------------
# LLM contract (mirrors llmkit; this module is standalone by design)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    # Present when the assistant asked for tools, or when this IS a tool result.
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"invalid role {self.role!r}; expected one of {ROLES}")


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


def tool_message(content: str, *, tool_call_id: str, name: str) -> Message:
    """A tool's OUTPUT, fed back to the model.

    Note the role: `tool`, not `user`. Providers treat tool results as a
    distinct role, and -- more importantly for safety -- so should you. Tool
    output is UNTRUSTED data, not instructions. See guardrails.py.
    """
    return Message("tool", content, tool_call_id=tool_call_id, name=name)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    """The model's request to run a tool. `arguments` is already parsed."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """Stable identity of (tool, arguments).

        Loop detection compares these: an agent calling the same tool with the
        same arguments twice in a row has learned nothing and will not learn
        anything on the third attempt either.
        """
        payload = json.dumps(self.arguments, sort_keys=True, default=str)
        return f"{self.name}({payload})"


@dataclass(frozen=True)
class ToolResult:
    """What a tool returned -- or how it failed.

    Failures are VALUES here, not exceptions. An agent must be able to see
    "that file does not exist" and try something else; an exception that
    unwinds the loop denies it the chance to recover, which is the whole point
    of giving it tools.
    """

    tool_call_id: str
    name: str
    content: str
    ok: bool = True
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def observation(self) -> str:
        """What the model actually sees."""
        if self.ok:
            return self.content
        # Phrased so the model can act on it. "Error" alone invites a retry;
        # naming the constraint invites a different approach.
        return f"ERROR: {self.error}"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = "fake-1"
    finish_reason: str = "stop"
    tool_calls: Sequence[ToolCall] = ()
    latency_ms: float = 0.0


@dataclass
class Step:
    """One iteration of the agent loop. The unit of the audit trail."""

    index: int
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None
    usage: Usage = field(default_factory=Usage)
    elapsed_ms: float = 0.0

    def render(self) -> str:
        lines = [f"[{self.index}] THOUGHT: {self.thought}" if self.thought else f"[{self.index}]"]
        if self.tool_call:
            lines.append(f"    ACTION: {self.tool_call.signature()}")
        if self.result:
            preview = self.result.observation.replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            lines.append(f"    OBSERVE: {preview}")
        return "\n".join(lines)


# Why a run ended. Only "final_answer" is unambiguous success -- and treating
# the others as success is how silent agent failures reach production.
STOP_REASONS = (
    "final_answer",
    "max_iterations",
    "budget_exceeded",
    "deadline_exceeded",
    "loop_detected",
    "no_progress",
    "blocked",
    "error",
)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    stop_reason: str = "final_answer"
    elapsed_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == "final_answer"

    @property
    def iterations(self) -> int:
        return len(self.steps)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool_call.name for s in self.steps if s.tool_call]

    def trace(self) -> str:
        """Human-readable trajectory. Print this when an agent misbehaves."""
        header = (
            f"stop_reason={self.stop_reason} steps={self.iterations} "
            f"tokens={self.usage.total_tokens} cost=${self.cost_usd:.6f}"
        )
        body = "\n".join(step.render() for step in self.steps)
        return f"{header}\n{'-' * len(header)}\n{body}\n=> {self.answer}"


# ---------------------------------------------------------------------------
# Offline provider
# ---------------------------------------------------------------------------


class FakeProvider:
    """Scriptable provider so agent trajectories are deterministic.

    Agents are the hardest thing in this repo to test, because the model
    chooses the control flow -- so an agent test with a real model is really a
    test of that model's mood today. Scripting the responses makes the LOOP
    testable: loop detection, budget enforcement, error recovery and
    termination are all properties of your code, not the model's.

    `respond_fn` receives the conversation so far and returns the next
    assistant message, which is enough to simulate any trajectory you like.
    """

    name = "fake"
    model = "fake-1"

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        *,
        respond_fn: Optional[Callable[[Sequence[Message]], str]] = None,
        tool_calls: Optional[Sequence[Optional[Sequence[ToolCall]]]] = None,
        latency: float = 0.0,
    ) -> None:
        self._responses = list(responses or [])
        self._respond_fn = respond_fn
        self._tool_calls = list(tool_calls or [])
        self.latency = latency
        self.calls = 0
        self.conversations: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        self.conversations.append(list(messages))
        if self.latency:
            time.sleep(self.latency)

        index = self.calls - 1
        if self._respond_fn is not None:
            text = self._respond_fn(messages)
        elif index < len(self._responses):
            text = self._responses[index]
        elif self._responses:
            text = self._responses[-1]  # repeat the last -- useful for loop tests
        else:
            digest = hashlib.sha256(str(messages[-1].content).encode()).hexdigest()[:8]
            text = f"[fake {digest}]"

        calls: Sequence[ToolCall] = ()
        if index < len(self._tool_calls) and self._tool_calls[index]:
            calls = tuple(self._tool_calls[index] or ())

        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=sum(len(m.content) for m in messages) // 4,
                output_tokens=max(1, len(text) // 4),
            ),
            tool_calls=calls,
            finish_reason="tool_use" if calls else "stop",
        )


class Timer:
    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self._started = 0.0

    def __enter__(self) -> Timer:
        self._started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._started) * 1000.0
