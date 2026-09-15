"""Embeddings: turning text into coordinates.

WHAT AN EMBEDDING ACTUALLY IS

A fixed-length vector of floats where *distance encodes meaning*. "How do I
reset my password?" and "I forgot my login" share no keywords, but land close
together. That is the entire trick behind semantic search, and the reason RAG
beats grep.

For a Java developer: it is a `double[1536]` that behaves like a very good
`hashCode()` -- except similar inputs produce NEARBY values instead of wildly
different ones, which is precisely what a hash is designed NOT to do.

THE THREE THINGS THAT BITE PEOPLE

1. **You cannot mix embedding models.** Vectors from `text-embedding-3-small`
   and `all-MiniLM-L6-v2` occupy different spaces; comparing them produces
   confident nonsense, not an error. Changing your embedding model means
   re-embedding the entire corpus. Store the model name with the vectors.

2. **Query and document must be embedded identically.** Same model, same
   normalization, same prefix convention. Some models (E5, BGE) REQUIRE
   asymmetric prefixes like "query: " / "passage: " and quietly underperform
   without them.

3. **Embedding is the expensive part of ingestion.** Cache by content hash.
   Re-embedding an unchanged corpus is the most common wasted spend in RAG.

DIMENSIONS

    384   all-MiniLM-L6-v2      fast, free, local, good enough surprisingly often
    768   all-mpnet-base-v2     better quality, still local
    1536  text-embedding-3-small   the sensible API default
    3072  text-embedding-3-large   marginal gains, 2x storage and RAM

Higher is not automatically better: 3072 floats x 4 bytes x 10M chunks is
~123 GB of index. Dimension is a cost decision as much as a quality one.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Any, Optional, Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    """Structural interface for anything that turns text into vectors."""

    name: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared by the hashing embedder and BM25."""
    return _TOKEN_RE.findall(text.lower())


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    Worth doing at write time: once every vector is unit-length, cosine
    similarity IS the dot product, so search becomes a single matrix multiply
    with no per-query division. Most vector databases do this internally.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid dividing by zero for empty strings.
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors, in [-1, 1].

    >>> round(cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 6)
    1.0
    >>> round(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 6)
    0.0
    """
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


class HashingEmbedder:
    """Deterministic, offline, dependency-free embeddings.

    Implements the "hashing trick": hash each token to a bucket and accumulate.
    No model, no download, no API key -- which is what makes every test and
    example in this module runnable anywhere, instantly and free.

    Be clear about what it is and is not. It captures LEXICAL overlap: documents
    sharing words land near each other. It does NOT capture semantics -- it has
    no idea "login" relates to "password". So it behaves like a fuzzy
    bag-of-words, which is enough to exercise and TEST the entire pipeline
    (chunking, storage, ranking, fusion, evaluation) but is not what you deploy.

    Swap in SentenceTransformerEmbedder or OpenAIEmbedder for real semantics;
    every other line of code stays the same. That substitutability is the point
    of the Embedder protocol.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 256, *, use_bigrams: bool = True) -> None:
        self.dimensions = dimensions
        self.use_bigrams = use_bigrams

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not security
        return int.from_bytes(digest[:4], "big") % self.dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts: dict[int, float] = {}
            for token in tokens:
                counts[self._bucket(token)] = counts.get(self._bucket(token), 0.0) + 1.0
            if self.use_bigrams:
                # Bigrams give a little word-order sensitivity, so that
                # "dog bites man" and "man bites dog" are not identical.
                for a, b in zip(tokens, tokens[1:]):
                    key = self._bucket(f"{a}_{b}")
                    counts[key] = counts.get(key, 0.0) + 0.5
            for bucket, count in counts.items():
                # Sublinear term frequency: the 10th occurrence of a word says
                # far less than the 1st. Same intuition as TF-IDF's log term.
                vectors[row, bucket] = 1.0 + math.log(count)
        return normalize(vectors)


class SentenceTransformerEmbedder:
    """Real local embeddings via sentence-transformers. No API cost.

    Optional dependency, imported lazily. `all-MiniLM-L6-v2` is 80 MB, runs on
    CPU at a few thousand sentences/second, and is genuinely competitive with
    paid APIs for English retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # optional
        except ImportError as exc:  # pragma: no cover - needs the optional dep
            raise ImportError(
                "pip install sentence-transformers (see requirements-optional.txt)"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return normalize(np.asarray(vectors, dtype=np.float32))


class OpenAIEmbedder:
    """Embeddings via the OpenAI HTTP API, using httpx directly.

    Batches aggressively: the API accepts many inputs per request, and one
    request per chunk is both slow and a fast route to a rate limit.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 128,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        self.dimensions = 3072 if "large" in model else 1536
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._client = client

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = self._client or httpx.Client(timeout=60.0)
        rows: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                rows.extend(item["embedding"] for item in response.json()["data"])
        finally:
            if self._client is None:
                client.close()
        return normalize(np.asarray(rows, dtype=np.float32))


class CachingEmbedder:
    """Wrap any embedder with a content-hash cache.

    The saving is not subtle. Re-ingesting a 50,000-chunk corpus where 200
    chunks changed costs 200 embeddings instead of 50,000 -- and on an API
    embedder that is the difference between cents and tens of dollars, every
    single ingest run.

    In production this cache belongs in Redis or a table keyed by
    (model_name, content_hash), not in process memory.
    """

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.dimensions = inner.dimensions
        self._cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def _key(self, text: str) -> str:
        # Model name is part of the key: vectors from different models are not
        # interchangeable, and serving one for the other is silent corruption.
        return hashlib.sha256(f"{self.inner.name}:{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._cache]

        if missing:
            fresh = self.inner.embed([t for _, t in missing])
            for (index, _), vector in zip(missing, fresh):
                self._cache[keys[index]] = vector

        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return np.vstack([self._cache[k] for k in keys]) if keys else np.zeros((0, self.dimensions))


def embed_chunks(embedder: Embedder, chunks: Sequence[Any], batch_size: int = 256) -> np.ndarray:
    """Embed chunk objects in batches.

    Batching matters for local models too: one forward pass over 256 sentences
    is dramatically faster than 256 passes over one, because it saturates the
    matrix multiply.
    """
    if not chunks:
        return np.zeros((0, embedder.dimensions), dtype=np.float32)
    out = [
        embedder.embed([c.text for c in chunks[i : i + batch_size]])
        for i in range(0, len(chunks), batch_size)
    ]
    return np.vstack(out)


class SemanticStubEmbedder:
    """A TEACHING DEVICE that imitates how a real dense model behaves.

    Not a real embedding model. It exists because `HashingEmbedder` is purely
    lexical, which makes it accidentally GOOD at exact identifiers -- the
    opposite of a real embedding model, and it therefore hides the entire
    argument for hybrid search.

    Real dense retrievers behave in two characteristic ways that this
    reproduces:

    1. **Synonyms collapse.** "leave", "holiday" and "vacation" land in the
       same region of space. A lexical matcher sees three unrelated tokens.
    2. **Rare identifiers blur.** `SKU-4471` is a low-frequency token that
       contributes almost nothing to a dense vector -- the model has no notion
       that the digits are decisive. It gets absorbed into a generic
       "error code" direction, so `SKU-4471` and `PAY-1180` look nearly
       identical. THIS is precisely why vector search fails on identifiers and
       why you pair it with BM25.

    Use it to demonstrate the tradeoff offline. Use a real model in anger.
    """

    name = "semantic-stub"

    # Words that mean the same thing are forced to share a bucket.
    SYNONYMS: dict[str, str] = {
        "holiday": "leave",
        "vacation": "leave",
        "pto": "leave",
        "annual": "leave",
        "reimbursement": "expense",
        "expenses": "expense",
        "claim": "expense",
        "refund": "money_back",
        "repayment": "money_back",
        "password": "credential",
        "login": "credential",
        "credentials": "credential",
        "secret": "credential",
        "key": "credential",
        "fault": "error",
        "failure": "error",
        "bug": "error",
        "issue": "error",
        "crash": "error",
        "deadline": "time_limit",
        "within": "time_limit",
        "days": "time_limit",
    }

    # Identifier fragments. NOTE: tokenize() splits on the hyphen, so "SKU-4471"
    # arrives as ["sku", "4471"] -- the discriminating part is the bare number.
    # Matching only "abc-1234" here would never fire, which is exactly the bug
    # that made an earlier version of this stub look identical to lexical search.
    _IDENTIFIER_RE = re.compile(r"^(?:[a-z]{2,4}-?\d{3,6}|\d{3,6})$")

    def __init__(self, dimensions: int = 256, *, identifier_weight: float = 0.05) -> None:
        self.dimensions = dimensions
        # Deliberately tiny: this is the "rare identifiers barely register" effect.
        self.identifier_weight = identifier_weight

    def _canonical(self, token: str) -> tuple[str, float]:
        if self._IDENTIFIER_RE.match(token):
            # Collapse every identifier onto ONE shared direction, and give it
            # almost no weight. Now SKU-4471 ~= PAY-1180, exactly as a real
            # dense model would see them.
            return "generic_identifier", self.identifier_weight
        return self.SYNONYMS.get(token, token), 1.0

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not security
        return int.from_bytes(digest[:4], "big") % self.dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                canonical, weight = self._canonical(token)
                vectors[row, self._bucket(canonical)] += weight
        return normalize(vectors)
