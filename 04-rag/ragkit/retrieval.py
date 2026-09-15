"""Retrieval strategies: keyword, vector, hybrid, diversity, and reranking.

THE LESSON OF THIS MODULE

Pure vector search is not the best retrieval strategy. It is the *default* one,
and it fails in a specific, predictable way: it is bad at exact terms.

Ask "what does error SKU-4471 mean?" and an embedding model has no idea that
`SKU-4471` is special -- it is a rare token that contributes little to a dense
vector. BM25, a keyword algorithm from the 1990s, nails it instantly.
Conversely ask "how do I stop the thing from crashing" and BM25 finds nothing
while the embedding model understands you.

    Dense (vector)   paraphrase, synonyms, fuzzy intent
    Sparse (BM25)    identifiers, error codes, names, acronyms, rare terms
    Hybrid           both, and it reliably beats either alone

Hybrid search is the single highest-value upgrade to a naive RAG system, and it
is maybe 40 lines of code. `examples/rag_ex_hybrid_vs_vector.py` shows the
numbers.

AFTER RETRIEVAL: RERANK

Retrieval optimizes for recall over thousands of chunks, cheaply. Reranking
optimizes for precision over the ~50 survivors, expensively. A cross-encoder
reads the query and document TOGETHER (rather than comparing two independently
computed vectors) and is dramatically more accurate -- far too slow for the
whole corpus, ideal for a shortlist.

    retrieve 50 cheaply  ->  rerank to 5 accurately  ->  generate
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .embeddings import Embedder, tokenize
from .types import Chunk, Message, RetrievalResult, ScoredChunk, Timer, user
from .vectorstore import MetadataFilter, VectorStore, dot


class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> RetrievalResult: ...


# ---------------------------------------------------------------------------
# Sparse: BM25
# ---------------------------------------------------------------------------


@dataclass
class BM25Retriever:
    """Okapi BM25, implemented from scratch.

    Still the strongest non-neural ranking function, and still competitive with
    embeddings on many benchmarks. Worth being able to write from memory.

        score(D, Q) = Σ  IDF(q) · ( f(q,D) · (k1 + 1) ) / ( f(q,D) + k1 · N )
                     q∈Q
        where N = 1 - b + b · |D| / avgdl

    Three ideas, each fixing a flaw in naive term counting:

    1. IDF -- a term appearing in every document tells you nothing. "the" is
       worthless; "SKU-4471" is decisive.
    2. Saturation (k1) -- the 20th occurrence of a word does not make a doc 20x
       more relevant. Term frequency saturates instead of growing linearly.
       This is the fix for keyword stuffing.
    3. Length normalization (b) -- long documents contain more of everything,
       so they would otherwise always win. Normalize by length vs the average.

    k1=1.5, b=0.75 are the standard defaults and rarely worth tuning.
    """

    chunks: list[Chunk] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self._docs: list[list[str]] = []
        self._freqs: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        if self.chunks:
            self.index(self.chunks)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._docs = [tokenize(c.text) for c in self.chunks]
        self._freqs = []
        self._df = {}
        for tokens in self._docs:
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self._freqs.append(freq)
            for token in freq:  # document frequency: once per doc, not per hit
                self._df[token] = self._df.get(token, 0) + 1
        self._avgdl = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency.

        The +0.5 terms are smoothing; the +1 inside the log keeps IDF positive
        for terms appearing in more than half the corpus (unsmoothed BM25 can
        go negative there, which lets a common term actively reduce a score).
        """
        n_docs = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], index: int) -> float:
        freq = self._freqs[index]
        doc_len = len(self._docs[index])
        total = 0.0
        for term in query_tokens:
            f = freq.get(term, 0)
            if not f:
                continue
            norm = 1 - self.b + self.b * (doc_len / self._avgdl if self._avgdl else 1.0)
            total += self.idf(term) * (f * (self.k1 + 1)) / (f + self.k1 * norm)
        return total

    def retrieve(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        with Timer() as timer:
            tokens = tokenize(query)
            candidates = [i for i, c in enumerate(self.chunks) if where is None or where(c)]
            scored = [(self.score(tokens, i), i) for i in candidates]
            scored = [(s, i) for s, i in scored if s > 0]
            scored.sort(key=lambda pair: -pair[0])
            top = [ScoredChunk(self.chunks[i], float(s), retriever="bm25") for s, i in scored[:k]]
        return RetrievalResult(query, top, timer.elapsed_ms)


# ---------------------------------------------------------------------------
# Dense: vector
# ---------------------------------------------------------------------------


@dataclass
class VectorRetriever:
    """Semantic search over a VectorStore."""

    store: VectorStore
    embedder: Embedder

    def retrieve(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        with Timer() as timer:
            vector = self.embedder.embed([query])[0]
            hits = self.store.search(vector, k=k, where=where)
        return RetrievalResult(query, hits, timer.elapsed_ms)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], *, k_constant: int = 60
) -> list[ScoredChunk]:
    """Merge ranked lists by RANK, ignoring the scores entirely.

        RRF(d) = Σ  1 / (k + rank_i(d))
                 i

    Why not just add the scores? Because BM25 scores are unbounded positives
    (0 to ~20) and cosine scores live in [-1, 1]. They are not commensurable,
    and normalizing them requires knowing each distribution, which shifts with
    every query. RRF sidesteps the problem: rank 1 is rank 1 in any system.

    k=60 is the value from the original paper (Cormack et al., 2009). It damps
    the difference between top ranks so one retriever cannot dominate purely by
    being confident.

    >>> a = [ScoredChunk(Chunk("x", chunk_id="1"), 9.0)]
    >>> b = [ScoredChunk(Chunk("y", chunk_id="2"), 0.9)]
    >>> [s.chunk.chunk_id for s in reciprocal_rank_fusion([a, b])]
    ['1', '2']
    """
    totals: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            key = scored.chunk.chunk_id
            totals[key] = totals.get(key, 0.0) + 1.0 / (k_constant + rank)
            # Keep the first sighting so we retain a representative chunk.
            best.setdefault(key, scored)

    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    return [ScoredChunk(best[key].chunk, score, retriever="hybrid-rrf") for key, score in ordered]


@dataclass
class HybridRetriever:
    """BM25 + vector, fused with RRF. The default you should reach for.

    Each retriever is asked for `fetch_k` (more than the final k) because
    fusion needs depth: a chunk ranked 8th by both systems should be able to
    beat one ranked 1st by a single system, and it cannot if you only fetched
    the top 3 from each.
    """

    vector: VectorRetriever
    bm25: BM25Retriever
    fetch_k: int = 20

    def retrieve(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        with Timer() as timer:
            dense = self.vector.retrieve(query, k=self.fetch_k, where=where).chunks
            sparse = self.bm25.retrieve(query, k=self.fetch_k, where=where).chunks
            fused = reciprocal_rank_fusion([dense, sparse])[:k]
        return RetrievalResult(query, fused, timer.elapsed_ms)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidates: Sequence[ScoredChunk],
    candidate_vectors: np.ndarray,
    *,
    k: int = 5,
    lambda_param: float = 0.5,
) -> list[ScoredChunk]:
    """Trade relevance against novelty.

        MMR = argmax [ λ · sim(d, Q) - (1-λ) · max sim(d, d_selected) ]

    The problem it solves is real and common: your top 5 chunks are five
    near-copies of the same paragraph, because a corpus with duplicated
    boilerplate will happily return all of it. The model then sees one fact
    five times and nothing else.

    λ=1.0 is pure relevance (no diversity), λ=0.0 is pure novelty (ignores the
    query). 0.5-0.7 is the useful range.
    """
    if not candidates:
        return []
    k = min(k, len(candidates))
    query = np.asarray(query_vector, dtype=np.float32).ravel()
    relevance = dot(candidate_vectors, query)

    selected: list[int] = [int(np.argmax(relevance))]
    while len(selected) < k:
        best_index, best_score = -1, -np.inf
        for i in range(len(candidates)):
            if i in selected:
                continue
            redundancy = max(
                float(dot(candidate_vectors[i], candidate_vectors[j])) for j in selected
            )
            score = lambda_param * float(relevance[i]) - (1 - lambda_param) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        if best_index < 0:
            break
        selected.append(best_index)

    return [
        ScoredChunk(candidates[i].chunk, float(relevance[i]), retriever="mmr") for i in selected
    ]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


@dataclass
class LLMReranker:
    """Score each candidate with an LLM. Accurate, slow, and expensive.

    Honest cost note: this is N extra LLM calls per query (or one call with all
    candidates, which is cheaper but less reliable). Use it when quality
    dominates cost -- legal, medical, expensive-decision retrieval -- and use a
    cross-encoder otherwise.
    """

    complete: Callable[[Sequence[Message]], Any]
    batch: bool = True

    _SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], k: int = 5
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scored: list[tuple[float, ScoredChunk]] = []
        for candidate in candidates:
            prompt = (
                "Rate how well the passage answers the question, 0-10.\n"
                "Reply with only a number.\n\n"
                f"Question: {query}\n\nPassage: {candidate.text[:1500]}\n\nScore:"
            )
            reply = self.complete([user(prompt)])
            match = self._SCORE_RE.search(getattr(reply, "text", str(reply)))
            score = float(match.group(1)) if match else 0.0
            scored.append((score, candidate))

        scored.sort(key=lambda pair: -pair[0])
        return [ScoredChunk(c.chunk, s, retriever="llm-rerank") for s, c in scored[:k]]


@dataclass
class CrossEncoderReranker:
    """Cross-encoder reranking via sentence-transformers. Optional dependency.

    Reads (query, passage) jointly rather than comparing two independent
    vectors, which is why it is far more accurate -- and why it cannot be
    precomputed, so it only works on a shortlist.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # optional
        except ImportError as exc:  # pragma: no cover - needs the optional dep
            raise ImportError("pip install sentence-transformers") from exc
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], k: int = 5
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.text) for c in candidates])
        order = np.argsort(-np.asarray(scores))[:k]
        return [
            ScoredChunk(candidates[i].chunk, float(scores[i]), retriever="cross-encoder")
            for i in order
        ]


# ---------------------------------------------------------------------------
# Query transformation
# ---------------------------------------------------------------------------


def multi_query(complete: Callable[[Sequence[Message]], Any], query: str, n: int = 3) -> list[str]:
    """Generate paraphrases and retrieve for all of them.

    Retrieval is sensitive to phrasing in ways users are not: "cancel my plan"
    and "how do I unsubscribe" can retrieve different chunks. Searching several
    phrasings and fusing the results reduces that variance.

    Cost: one extra LLM call, plus N retrievals. Usually worth it.
    """
    prompt = (
        f"Write {n} alternative phrasings of this search query. "
        "One per line, no numbering, no commentary.\n\n"
        f"Query: {query}"
    )
    reply = complete([user(prompt)])
    text = getattr(reply, "text", str(reply))
    variants = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    # Always keep the original: a paraphrase can drift off-meaning.
    return [query, *variants[:n]]


def hyde(complete: Callable[[Sequence[Message]], Any], query: str) -> str:
    """HyDE -- Hypothetical Document Embeddings.

    Ask the model to INVENT an answer, then embed that instead of the question.

    The insight: a question and its answer are lexically and structurally very
    different ("What is the refund window?" vs "Refunds are accepted within 30
    days of purchase"). Embedding similarity compares a question to documents
    that are answers. A hypothetical answer -- even a factually wrong one --
    sits much closer in vector space to the real answer.

    The hallucinated content does not matter: it is never shown to anyone, it
    only serves as a better query vector.
    """
    reply = complete(
        [user(f"Write a short factual passage that would answer this question.\n\n{query}")]
    )
    return getattr(reply, "text", str(reply))


@dataclass
class ParentChildRetriever:
    """Retrieve on small chunks, return their larger parents. "Small-to-big".

    Resolves the core chunking tension directly instead of compromising on a
    middle size: small chunks match precisely (focused embeddings), large
    chunks answer completely (full context). Search the children, hand the
    model the parents.
    """

    child_retriever: Any
    parents: dict[str, Chunk]  # parent_id -> parent chunk

    def retrieve(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        with Timer() as timer:
            children = self.child_retriever.retrieve(query, k=k * 3, where=where).chunks
            seen: dict[str, ScoredChunk] = {}
            for child in children:
                parent_id = child.chunk.metadata.get("parent_id", child.chunk.chunk_id)
                parent = self.parents.get(parent_id, child.chunk)
                # Deduplicate: several children often share one parent, and the
                # best child's score represents the parent.
                if parent_id not in seen:
                    seen[parent_id] = ScoredChunk(parent, child.score, retriever="parent-child")
                if len(seen) >= k:
                    break
        return RetrievalResult(query, list(seen.values()), timer.elapsed_ms)
