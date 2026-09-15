"""Core types for the RAG pipeline, plus a standalone offline provider.

This module is deliberately self-contained -- it re-declares the LLM contract
from `03-llm-integration/llmkit` rather than importing it. Two reasons:

1. Directory names starting with a digit (`03-llm-integration`) are not valid
   Python identifiers, so cross-module imports would need packaging ceremony
   that distracts from the subject.
2. Each module in this curriculum must be studiable on its own.

The field names are IDENTICAL across modules on purpose, so the repo reads as
one system rather than five dialects.

THE VOCABULARY OF RAG

    Document   what you ingested        (a whole file)
    Chunk      what you retrieve        (a slice of a document)
    Embedding  a chunk as a vector      (meaning, as coordinates)
    ScoredChunk  a chunk + its relevance to one specific query

The distinction between Document and Chunk is the one beginners collapse, and
it is the source of most bad RAG. You embed and retrieve CHUNKS; you cite
DOCUMENTS. Chunk metadata is what lets you get back from one to the other.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# LLM contract (mirrors llmkit; see module docstring)
# ---------------------------------------------------------------------------

ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"invalid role {self.role!r}")


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


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


class FakeProvider:
    """Deterministic offline provider, so the whole RAG pipeline is testable.

    `answer_fn` lets a test control generation while still exercising the real
    retrieval path -- which is where RAG bugs actually live.
    """

    name = "fake"
    model = "fake-1"

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        *,
        answer_fn: Optional[Any] = None,
    ) -> None:
        self._responses = list(responses or [])
        self._answer_fn = answer_fn
        self.calls = 0
        self.prompts: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        self.prompts.append(list(messages))
        if self._answer_fn is not None:
            text = self._answer_fn(messages)
        elif self._responses:
            text = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        else:
            last = messages[-1].content if messages else ""
            digest = hashlib.sha256(last.encode()).hexdigest()[:8]
            text = f"[fake answer {digest}]"
        return LLMResponse(
            text=text,
            usage=Usage(sum(len(m.content) for m in messages) // 4, max(1, len(text) // 4)),
        )


# ---------------------------------------------------------------------------
# RAG types
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A whole source file, before chunking."""

    text: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = self.content_hash[:16]

    @property
    def content_hash(self) -> str:
        """Hash of the CONTENT, not the path.

        This is what makes incremental re-indexing possible: on re-ingest, a
        document whose hash is unchanged does not need re-chunking or
        re-embedding. Skipping that is often the single biggest cost saving in
        a production RAG system -- re-embedding an unchanged corpus nightly is
        a classic way to burn money for nothing.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """A retrievable slice of a document.

    `start`/`end` are character offsets into the parent document, which makes a
    chunk traceable back to its exact source span -- needed for highlighting
    and for verifying a citation is not fabricated.
    """

    text: str
    doc_id: str = ""
    source: str = "unknown"
    ordinal: int = 0  # position within the parent document
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raw = f"{self.doc_id}:{self.ordinal}:{self.text[:64]}"
            self.chunk_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        if self.end == 0:
            self.end = self.start + len(self.text)

    def __len__(self) -> int:
        return len(self.text)

    @property
    def heading_path(self) -> str:
        """Breadcrumb like 'Handbook > Benefits > Leave', when known.

        Structure-aware chunkers populate this. It matters more than it looks:
        a chunk reading "employees get 25 days" is ambiguous on its own, and
        prefixing the heading path is a cheap, large retrieval-quality win.
        """
        return self.metadata.get("heading_path", "")


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its relevance to one particular query."""

    chunk: Chunk
    score: float
    # Which retriever produced it; hybrid search needs this for diagnostics.
    retriever: str = "unknown"

    @property
    def text(self) -> str:
        return self.chunk.text

    def __lt__(self, other: ScoredChunk) -> bool:
        return self.score < other.score


@dataclass
class RetrievalResult:
    """Everything a retrieval produced, with the timing to debug it."""

    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def __iter__(self) -> Iterator[ScoredChunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def sources(self) -> list[str]:
        """Unique source documents, order preserved."""
        seen: list[str] = []
        for scored in self.chunks:
            if scored.chunk.source not in seen:
                seen.append(scored.chunk.source)
        return seen


@dataclass
class Answer:
    """A generated answer plus the evidence it was built from.

    Carrying `chunks` alongside `text` is not decoration: without the evidence
    you cannot verify a citation, debug a bad answer, or compute faithfulness.
    An answer without its sources is unfalsifiable.
    """

    text: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    elapsed_ms: float = 0.0
    refused: bool = False  # True when retrieval was too weak to answer

    @property
    def citations(self) -> list[str]:
        seen: list[str] = []
        for scored in self.chunks:
            if scored.chunk.source not in seen:
                seen.append(scored.chunk.source)
        return seen


class Timer:
    """Tiny context manager for the elapsed_ms fields above."""

    def __init__(self) -> None:
        self.started = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic() - self.started) * 1000.0
