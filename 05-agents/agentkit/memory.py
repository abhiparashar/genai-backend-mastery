"""Agent memory: what to keep, what to summarise, what to throw away.

THE PROBLEM

Every LLM call is stateless. "Memory" is just you deciding which messages to
resend, and that decision is a budget problem with a correctness constraint.

An agent run grows its own context: each step adds an assistant message and a
tool observation. Ten steps of a tool returning 2KB of JSON is 20KB of context
before the user has said anything twice. Three things then go wrong, in order:

    1. cost     -- you resend the entire history on EVERY turn, so an N-turn
                   conversation costs O(N^2) tokens, not O(N)
    2. failure  -- eventually you exceed the context window and the call errors
    3. accuracy -- before that, quality degrades: models attend less reliably
                   to the middle of a long context ("lost in the middle")

THE STRATEGIES

    BufferMemory    keep everything. Correct until it isn't. Fine for short tasks.
    WindowMemory    keep the last N. Cheap, predictable, forgets abruptly.
    SummaryMemory   compress old turns into a summary. Keeps the gist, loses
                    detail, and costs an extra LLM call to produce.
    VectorMemory    store everything, retrieve only what is relevant now.
                    Scales indefinitely; retrieval can miss.

THE RULE THAT APPLIES TO ALL OF THEM

**Never drop the system message.** It carries the agent's instructions and
tool contract. A window that naively keeps "the last 10 messages" will
eventually evict it, and the agent then forgets what it is and what it may
call -- a spectacular and confusing failure. Every implementation here pins it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .types import Message, assistant, system, user

CHARS_PER_TOKEN = 4


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Rough token count, including per-message chat overhead."""
    return sum(len(m.content) // CHARS_PER_TOKEN + 4 for m in messages)


class Memory(Protocol):
    def add(self, message: Message) -> None: ...

    def messages(self) -> list[Message]: ...

    def clear(self) -> None: ...


@dataclass
class BufferMemory:
    """Keep every message. The honest baseline.

    Correct and simple; the cost is quadratic in turns. Use it for short,
    bounded tasks -- and know the ceiling you are heading for.
    """

    history: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)

    def messages(self) -> list[Message]:
        return list(self.history)

    def clear(self) -> None:
        self.history.clear()

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.history)


@dataclass
class WindowMemory:
    """Keep the system message plus the last `k` turns.

    Predictable cost, abrupt forgetting: the agent will cheerfully repeat a
    question answered 12 messages ago. Prefer SummaryMemory when the early
    context carries decisions rather than chatter.
    """

    k: int = 10
    history: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)

    def messages(self) -> list[Message]:
        pinned = [m for m in self.history if m.role == "system"]
        rest = [m for m in self.history if m.role != "system"]
        return pinned + rest[-self.k :]

    def clear(self) -> None:
        self.history.clear()


@dataclass
class TokenWindowMemory:
    """Trim to a TOKEN budget rather than a message count.

    Better than WindowMemory in practice, because messages vary enormously in
    size: ten one-line turns and ten 4KB tool outputs are the same `k` and a
    40x difference in cost. Budgets are denominated in tokens, so trim in
    tokens.

    Drops oldest-first and always pins the system message.
    """

    max_tokens: int = 4000
    history: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)

    def messages(self) -> list[Message]:
        pinned = [m for m in self.history if m.role == "system"]
        rest = [m for m in self.history if m.role != "system"]

        budget = self.max_tokens - estimate_tokens(pinned)
        kept: list[Message] = []
        for message in reversed(rest):  # newest first: recency wins
            cost = len(message.content) // CHARS_PER_TOKEN + 4
            if cost > budget:
                break
            kept.append(message)
            budget -= cost
        kept.reverse()
        return pinned + kept

    def clear(self) -> None:
        self.history.clear()

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.messages())


@dataclass
class SummaryMemory:
    """Summarise older turns once the history crosses a threshold.

    Keeps the gist of a long conversation at bounded cost. Two honest caveats:

    - summarising costs an extra LLM call, so it is not free
    - summaries lose detail irreversibly, and you cannot know in advance which
      detail mattered. Never summarise something you must be able to quote.

    `summarize_fn` is injected so this is testable offline.
    """

    summarize_fn: Callable[[Sequence[Message]], str]
    threshold_tokens: int = 2000
    keep_recent: int = 4
    history: list[Message] = field(default_factory=list)
    summary: str = ""
    summarizations: int = 0

    def add(self, message: Message) -> None:
        self.history.append(message)
        self._maybe_compress()

    def _maybe_compress(self) -> None:
        rest = [m for m in self.history if m.role != "system"]
        if estimate_tokens(rest) <= self.threshold_tokens:
            return

        older, recent = rest[: -self.keep_recent], rest[-self.keep_recent :]
        if not older:
            return

        # Fold the previous summary in, so nothing is silently dropped when
        # compression happens repeatedly.
        to_summarize = list(older)
        if self.summary:
            to_summarize.insert(0, assistant(f"Summary of earlier conversation: {self.summary}"))

        self.summary = self.summarize_fn(to_summarize)
        self.summarizations += 1
        pinned = [m for m in self.history if m.role == "system"]
        self.history = pinned + recent

    def messages(self) -> list[Message]:
        pinned = [m for m in self.history if m.role == "system"]
        rest = [m for m in self.history if m.role != "system"]
        if self.summary:
            return [*pinned, assistant(f"Summary of earlier conversation: {self.summary}"), *rest]
        return pinned + rest

    def clear(self) -> None:
        self.history.clear()
        self.summary = ""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _hash_embed(text: str, dimensions: int = 128) -> list[float]:
    """Dependency-free embedding for offline semantic recall."""
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode()).digest()  # noqa: S324 - not security
        vector[int.from_bytes(digest[:4], "big") % dimensions] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


@dataclass
class VectorMemory:
    """Store everything; retrieve only what is relevant to the current turn.

    The only strategy that scales indefinitely, because context size stops
    depending on conversation length. It is also the only one that can fail
    silently: if retrieval misses, the agent behaves as though the fact was
    never mentioned, which looks like amnesia rather than an error.

    Use it for long-lived assistants. Use TokenWindowMemory for a single task.
    """

    top_k: int = 4
    dimensions: int = 128
    history: list[Message] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)
        self._vectors.append(_hash_embed(message.content, self.dimensions))

    def recall(self, query: str) -> list[Message]:
        if not self.history:
            return []
        query_vector = _hash_embed(query, self.dimensions)
        scored = [
            (sum(a * b for a, b in zip(query_vector, vector)), index)
            for index, vector in enumerate(self._vectors)
            if self.history[index].role != "system"
        ]
        scored.sort(reverse=True)
        chosen = sorted(index for score, index in scored[: self.top_k] if score > 0)
        return [self.history[i] for i in chosen]

    def messages(self, query: str = "") -> list[Message]:
        pinned = [m for m in self.history if m.role == "system"]
        if not query:
            return pinned + [m for m in self.history if m.role != "system"][-self.top_k :]
        return pinned + self.recall(query)

    def clear(self) -> None:
        self.history.clear()
        self._vectors.clear()


@dataclass
class ConversationBudget:
    """Enforce a token ceiling on any message list.

    The last line of defence before a context-length error. Unlike the memory
    classes this is stateless -- hand it messages, get back messages that fit.

    Truncates oversized individual messages rather than dropping them, because
    a single 50KB tool output should not evict the entire conversation.
    """

    max_tokens: int = 8000
    max_message_tokens: int = 2000

    def apply(self, messages: Sequence[Message]) -> list[Message]:
        capped: list[Message] = []
        for message in messages:
            limit = self.max_message_tokens * CHARS_PER_TOKEN
            if len(message.content) > limit:
                capped.append(
                    Message(
                        message.role,
                        message.content[:limit] + "\n...[truncated]",
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                    )
                )
            else:
                capped.append(message)

        pinned = [m for m in capped if m.role == "system"]
        rest = [m for m in capped if m.role != "system"]
        budget = self.max_tokens - estimate_tokens(pinned)

        kept: list[Message] = []
        for message in reversed(rest):
            cost = len(message.content) // CHARS_PER_TOKEN + 4
            if cost > budget:
                break
            kept.append(message)
            budget -= cost
        kept.reverse()
        return pinned + kept


def default_summarizer(provider: Any) -> Callable[[Sequence[Message]], str]:
    """Build a summarizer backed by an LLM.

    The prompt asks for decisions and facts rather than a narrative, because
    "the user asked about leave, then about expenses" is useless to the agent,
    while "user is a permanent employee in Berlin; leave balance 12 days" is
    what it actually needs to keep behaving correctly.
    """

    def summarize(messages: Sequence[Message]) -> str:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        response = provider.complete(
            [
                system(
                    "Summarise this conversation for an assistant that must continue it. "
                    "Record concrete facts, decisions, and unresolved questions. "
                    "Omit pleasantries and narration. Be terse."
                ),
                user(transcript),
            ]
        )
        return response.text.strip()

    return summarize


__all__ = [
    "Memory",
    "BufferMemory",
    "WindowMemory",
    "TokenWindowMemory",
    "SummaryMemory",
    "VectorMemory",
    "ConversationBudget",
    "default_summarizer",
    "estimate_tokens",
]
