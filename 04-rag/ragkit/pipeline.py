"""The end-to-end RAG pipeline: ingest -> retrieve -> ground -> generate.

    ingest:  documents -> chunk -> embed -> store   (+ BM25 index)
    query:   question -> retrieve -> rerank -> budget -> prompt -> answer

THE THREE THINGS THAT MAKE THIS PRODUCTION-GRADE

1. **Citations that are verifiable.** Every claim maps to a numbered source the
   user can open. An LLM answer without provenance is unfalsifiable, and
   unfalsifiable answers are how RAG systems quietly ship wrong information.

2. **A context budget.** Retrieved chunks get TRUNCATED to fit a token budget,
   deterministically. Without this you eventually send a 200k-token prompt and
   get a context-length error in production at 3am -- or worse, silently pay
   10x for a prompt stuffed with marginal chunks.

3. **A refusal path.** When the best retrieved chunk scores below a threshold,
   the correct answer is "I don't know", NOT a fluent guess. This is the single
   highest-value guardrail in RAG: a system that admits ignorance is trusted;
   one that hallucinates confidently is abandoned after the first bad answer.

WHY "MORE CONTEXT" IS NOT FREE

It is tempting to send 20 chunks and let the model sort it out. Three reasons
not to: cost scales linearly with input tokens; latency scales with them too;
and accuracy actually DEGRADES -- models attend less reliably to information
buried in the middle of a long context ("lost in the middle"). Five good chunks
beat twenty mediocre ones on every axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .chunking import Chunker, RecursiveCharacterChunker
from .embeddings import CachingEmbedder, Embedder, HashingEmbedder
from .retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from .types import (
    Answer,
    Chunk,
    Document,
    LLMResponse,
    Message,
    RetrievalResult,
    ScoredChunk,
    Timer,
    Usage,
    system,
    user,
)
from .vectorstore import InMemoryVectorStore, MetadataFilter, dot

CHARS_PER_TOKEN = 4

# Deliberately explicit: the model is told to refuse rather than improvise.
# "If the context does not contain the answer, say so" is the most
# cost-effective hallucination control available.
DEFAULT_SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.

Rules:
- Cite every factual claim with its source number, like [1] or [2].
- If the sources do not contain the answer, reply exactly: I don't know based on the provided sources.
- Never use knowledge outside the sources, even if you are confident.
- Be concise."""


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], k: int = 5
    ) -> list[ScoredChunk]: ...


class Completer(Protocol):
    def complete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse: ...


def fit_to_budget(
    chunks: Sequence[ScoredChunk], max_tokens: int, *, min_chunk_tokens: int = 50
) -> list[ScoredChunk]:
    """Take chunks in order until the token budget is exhausted.

    Chunks arrive ranked, so this keeps the best ones. A chunk that does not
    fit is skipped rather than truncated mid-sentence -- a half-sentence is
    worse than no sentence, because the model will confidently complete it.

    We keep scanning after a skip: a later, smaller chunk may still fit, and
    dropping it would waste budget for no reason.
    """
    out: list[ScoredChunk] = []
    used = 0
    for scored in chunks:
        needed = max(1, len(scored.text) // CHARS_PER_TOKEN)
        if used + needed <= max_tokens:
            out.append(scored)
            used += needed
        elif max_tokens - used >= min_chunk_tokens:
            continue  # room remains; a smaller later chunk may still fit
        else:
            break
    return out


def format_context(chunks: Sequence[ScoredChunk]) -> str:
    """Render chunks as numbered sources the model can cite.

    The numbering is the contract: `[1]` in the answer must map to sources[0].
    Including the heading path and source filename measurably improves both
    retrieval grounding and the user's ability to verify.
    """
    blocks = []
    for index, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        label = chunk.heading_path or chunk.source
        blocks.append(f"[{index}] ({label})\n{chunk.text}")
    return "\n\n".join(blocks)


@dataclass
class RAGPipeline:
    """A complete, configurable RAG system.

    Defaults are chosen to run offline with zero dependencies: a hashing
    embedder and an in-memory store. Swap in real components without touching
    query-time code -- that substitutability is the reason everything is a
    Protocol.
    """

    provider: Completer
    embedder: Embedder = field(default_factory=lambda: CachingEmbedder(HashingEmbedder(256)))
    chunker: Chunker = field(default_factory=lambda: RecursiveCharacterChunker(1200, 150))
    reranker: Optional[Reranker] = None

    top_k: int = 5
    fetch_k: int = 20
    max_context_tokens: int = 2000
    # Below this top score, refuse rather than guess. Tune against a golden
    # set -- too high and you refuse answerable questions, too low and you
    # hallucinate. Measure, do not guess.
    min_score: float = 0.05
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_hybrid: bool = True

    def __post_init__(self) -> None:
        self.store = InMemoryVectorStore(self.embedder.dimensions)
        self.bm25 = BM25Retriever()
        self.chunks: list[Chunk] = []
        self._doc_hashes: dict[str, str] = {}

    # -- ingestion ---------------------------------------------------------

    def ingest(self, documents: Sequence[Document]) -> dict[str, int]:
        """Chunk, embed and index documents.

        Skips documents whose content hash is unchanged. This is what makes
        re-ingestion cheap, and it is the difference between a nightly job
        that costs cents and one that costs hundreds of dollars.
        """
        new_docs = [d for d in documents if self._doc_hashes.get(d.source) != d.content_hash]
        skipped = len(documents) - len(new_docs)
        if not new_docs:
            return {"documents": 0, "chunks": 0, "skipped": skipped}

        fresh: list[Chunk] = []
        for document in new_docs:
            fresh.extend(self.chunker.split(document))
            self._doc_hashes[document.source] = document.content_hash

        if fresh:
            vectors = self.embedder.embed([c.text for c in fresh])
            self.store.add(fresh, vectors)
            self.chunks.extend(fresh)
            # BM25 keeps its own inverted index, so it must be rebuilt.
            self.bm25.index(self.chunks)

        return {"documents": len(new_docs), "chunks": len(fresh), "skipped": skipped}

    # -- retrieval ---------------------------------------------------------

    def _retriever(self):
        vector = VectorRetriever(self.store, self.embedder)
        if not self.use_hybrid:
            return vector
        return HybridRetriever(vector, self.bm25, fetch_k=self.fetch_k)

    def retrieve(self, question: str, *, where: Optional[MetadataFilter] = None) -> RetrievalResult:
        # Over-fetch when reranking: the reranker needs candidates to choose
        # among, and its whole value is reordering a shortlist.
        k = self.fetch_k if self.reranker else self.top_k
        result = self._retriever().retrieve(question, k=k, where=where)
        if self.reranker and result.chunks:
            result.chunks = self.reranker.rerank(question, result.chunks, k=self.top_k)
        else:
            result.chunks = result.chunks[: self.top_k]
        return result

    def relevance(self, question: str, chunks: Sequence[ScoredChunk]) -> float:
        """Best TRUE cosine similarity between the query and the retrieved chunks.

        The refusal gate cannot use `RetrievalResult.top_score`, because that
        score means different things per retriever: cosine is [-1, 1], BM25 is
        unbounded, and RRF is ~1/(60+rank) which maxes out around 0.016. A
        single threshold compared against RRF scores would refuse EVERY query
        -- which is exactly the bug this method exists to prevent.

        So we recover a comparable number: embed the query once and take the
        best cosine against the stored chunk vectors.
        """
        if not chunks:
            return 0.0
        query_vector = self.embedder.embed([question])[0]
        best = 0.0
        for scored in chunks:
            vector = self.store.vector_for(scored.chunk.chunk_id)
            if vector is None:
                continue
            best = max(best, float(dot(vector, query_vector)))
        return best

    # -- generation --------------------------------------------------------

    def answer(self, question: str, *, where: Optional[MetadataFilter] = None) -> Answer:
        with Timer() as timer:
            retrieved = self.retrieve(question, where=where)

            # Judge on true cosine, NOT retrieved.top_score - see relevance().
            score = self.relevance(question, retrieved.chunks)
            if not retrieved.chunks or score < self.min_score:
                # Refuse BEFORE spending a token. A guess here is worse than
                # silence, and it is also more expensive.
                return Answer(
                    text="I don't know based on the provided sources.",
                    chunks=list(retrieved.chunks),
                    elapsed_ms=timer.elapsed_ms,
                    refused=True,
                )

            kept = fit_to_budget(retrieved.chunks, self.max_context_tokens)
            prompt = (
                f"Sources:\n\n{format_context(kept)}\n\n"
                f"Question: {question}\n\n"
                "Answer using only the sources above, citing them by number."
            )
            response = self.provider.complete([system(self.system_prompt), user(prompt)])

        return Answer(
            text=response.text,
            chunks=kept,
            usage=response.usage,
            elapsed_ms=timer.elapsed_ms,
        )

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        hit_rate = getattr(self.embedder, "hit_rate", None)
        return {
            "documents": len(self._doc_hashes),
            "chunks": len(self.chunks),
            "vectors": len(self.store),
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
            "embed_cache_hit_rate": hit_rate,
        }


def build_pipeline(provider: Completer, **kwargs: Any) -> RAGPipeline:
    """Convenience constructor with offline-friendly defaults."""
    return RAGPipeline(provider=provider, **kwargs)


def citation_check(answer: Answer) -> dict[str, Any]:
    """Verify the answer's citations refer to sources that were actually sent.

    Catches a real and nasty failure: a model emitting [7] when only 5 sources
    were provided. Users click through, find nothing, and stop trusting every
    citation -- including the correct ones.
    """
    import re

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer.text)}
    available = set(range(1, len(answer.chunks) + 1))
    dangling = sorted(cited - available)
    return {
        "cited": sorted(cited),
        "available": sorted(available),
        "dangling": dangling,
        "valid": not dangling,
        "uncited_sources": sorted(available - cited),
        "has_citations": bool(cited),
    }


def estimate_prompt_tokens(chunks: Sequence[ScoredChunk], question: str) -> int:
    body = sum(len(c.text) for c in chunks) + len(question) + len(DEFAULT_SYSTEM_PROMPT)
    return body // CHARS_PER_TOKEN


__all__ = [
    "RAGPipeline",
    "build_pipeline",
    "fit_to_budget",
    "format_context",
    "citation_check",
    "estimate_prompt_tokens",
    "DEFAULT_SYSTEM_PROMPT",
    "Answer",
    "Usage",
]
