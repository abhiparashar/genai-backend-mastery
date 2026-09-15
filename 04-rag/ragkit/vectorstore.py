"""Vector storage and similarity search.

WHAT A VECTOR DATABASE IS FOR

Finding the nearest vectors to a query vector. That is genuinely all it does.
With 10,000 chunks you do not need one -- numpy does an exact search in
milliseconds. You need one when:

- the corpus outgrows RAM (roughly >1M vectors),
- you need durability, concurrent writes, or replication,
- you need metadata filtering at scale,
- or you need sub-linear search because exact scan has become too slow.

Do not start with Pinecone. Start with numpy, move to pgvector when you need
persistence (you almost certainly already run Postgres), and only adopt a
dedicated vector DB when you can state which of the above forced you.

EXACT VS APPROXIMATE

Exact search compares the query to every vector: O(N·D), perfect recall.
At 1M x 1536 floats that is ~6 GB and ~100 ms per query -- too slow for an
interactive endpoint.

ANN (approximate nearest neighbour) trades a little recall for a lot of speed,
typically 95-99% recall at 10-100x the throughput. `IVFIndex` below implements
the clustering idea behind FAISS's IVF so the tradeoff is concrete and
measurable rather than a vendor claim.

THE FILTERING TRAP (pre- vs post-filter)

Post-filtering searches the whole corpus, then discards results failing the
filter. Ask for top-10 from tenant A and you may get zero back, because all 10
nearest vectors belonged to tenant B. Pre-filtering restricts the candidate set
FIRST and always returns 10. This is not an optimization; it is a correctness
issue, and in multi-tenant systems it is also a data-isolation issue.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Optional, Protocol

import numpy as np

from .types import Chunk, ScoredChunk

MetadataFilter = Callable[[Chunk], bool]


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply with spurious FP warnings suppressed.

    numpy 2.0.x built against Apple's Accelerate BLAS emits bogus
    "divide by zero"/"overflow encountered in matmul" RuntimeWarnings on
    perfectly finite input. Verified against plain numpy with clean unit
    vectors: no NaNs, no infs, all norms exactly 1.0, and correct results --
    the warning comes from the vectorized kernel, not the data.

    Suppressed HERE rather than with a global filter, so a genuine numerical
    problem anywhere else in the codebase still surfaces.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, query_vector: np.ndarray, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> list[ScoredChunk]: ...

    def __len__(self) -> int: ...


def tenant_filter(tenant: str) -> MetadataFilter:
    """Filter chunks to one tenant.

    Reach for this reflexively in any multi-tenant system. The failure mode --
    tenant A retrieving tenant B's document text into an LLM prompt -- is a
    data breach, not a bug, and it is invisible in testing unless you
    explicitly test for it.
    """

    def _filter(chunk: Chunk) -> bool:
        return chunk.metadata.get("tenant") == tenant

    return _filter


class InMemoryVectorStore:
    """Exact cosine search over a numpy matrix.

    Correct, simple, and the right choice far longer than people assume. It is
    also the reference implementation that `IVFIndex` is measured against.
    """

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.chunks: list[Chunk] = []
        self._vectors: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._vectors

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dimensions:
            # Catching this here converts a silent quality collapse into a
            # loud error. Mixing embedding models is otherwise undetectable.
            raise ValueError(
                f"dimension mismatch: store is {self.dimensions}, got {vectors.shape[1]}. "
                "Are you mixing embedding models?"
            )
        self.chunks.extend(chunks)
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        *,
        where: Optional[MetadataFilter] = None,
    ) -> list[ScoredChunk]:
        if not self.chunks:
            return []

        # PRE-filter: restrict candidates before ranking, so k results are
        # actually returned. See the module docstring.
        candidates = (
            list(range(len(self.chunks)))
            if where is None
            else [i for i, c in enumerate(self.chunks) if where(c)]
        )
        if not candidates:
            return []

        matrix = self.vectors[candidates]
        query = np.asarray(query_vector, dtype=np.float32).ravel()

        # Vectors are unit-normalized at write time, so the dot product IS
        # cosine similarity -- one matrix multiply, no per-row division.
        scores = dot(matrix, query)

        k = min(k, len(candidates))
        # argpartition is O(N) vs argsort's O(N log N); we only sort the k we keep.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            ScoredChunk(self.chunks[candidates[i]], float(scores[i]), retriever="vector")
            for i in top
        ]

    def delete(self, predicate: MetadataFilter) -> int:
        """Remove matching chunks. Returns how many went.

        Needed more often than people plan for: GDPR deletion, a document being
        unpublished, or re-indexing a changed file without duplicating it.
        """
        keep = [i for i, c in enumerate(self.chunks) if not predicate(c)]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self._vectors = self.vectors[keep] if keep else None
        return removed


class IVFIndex:
    """Approximate search by inverted file / coarse clustering.

    The idea behind FAISS's IVF, in ~40 lines:

        BUILD   cluster all vectors into `n_lists` groups (k-means)
        SEARCH  find the `n_probe` nearest centroids, then search ONLY the
                vectors in those clusters

    With 1M vectors in 1000 clusters and n_probe=10, you compare against ~1%
    of the corpus. That is the speedup. The cost is recall: a true nearest
    neighbour sitting just across a cluster boundary is missed.

    `n_probe` is the dial. n_probe=1 is fastest and least accurate;
    n_probe=n_lists degenerates to exact search. Tune it by MEASURING recall
    against the exact store -- `examples/` does exactly that.
    """

    def __init__(
        self, dimensions: int, *, n_lists: int = 16, n_probe: int = 3, seed: int = 0
    ) -> None:
        self.dimensions = dimensions
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.seed = seed
        self.chunks: list[Chunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._assignments: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        self.chunks.extend(chunks)
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._centroids = None  # index is stale; rebuild lazily on next search

    def _build(self) -> None:
        """Lloyd's algorithm k-means. Deliberately small and readable."""
        vectors = self._vectors
        assert vectors is not None
        n_lists = min(self.n_lists, len(vectors))

        rng = np.random.default_rng(self.seed)
        centroids = vectors[rng.choice(len(vectors), size=n_lists, replace=False)].copy()

        for _ in range(10):
            assignments = np.argmax(dot(vectors, centroids.T), axis=1)
            moved = False
            for c in range(n_lists):
                members = vectors[assignments == c]
                if len(members) == 0:
                    continue
                new_centroid = members.mean(axis=0)
                norm = np.linalg.norm(new_centroid)
                if norm:
                    new_centroid = new_centroid / norm
                if not np.allclose(new_centroid, centroids[c]):
                    centroids[c] = new_centroid
                    moved = True
            if not moved:
                break  # converged

        self._centroids = centroids
        self._assignments = np.argmax(dot(vectors, centroids.T), axis=1)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        *,
        where: Optional[MetadataFilter] = None,
    ) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        if self._centroids is None:
            self._build()
        assert self._centroids is not None and self._assignments is not None

        query = np.asarray(query_vector, dtype=np.float32).ravel()

        n_probe = min(self.n_probe, len(self._centroids))
        centroid_scores = dot(self._centroids, query)
        probes = np.argsort(-centroid_scores)[:n_probe]

        candidates = np.where(np.isin(self._assignments, probes))[0]
        if where is not None:
            candidates = np.array([i for i in candidates if where(self.chunks[i])], dtype=int)
        if candidates.size == 0:
            return []

        assert self._vectors is not None
        scores = dot(self._vectors[candidates], query)
        k = min(k, len(candidates))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            ScoredChunk(self.chunks[candidates[i]], float(scores[i]), retriever="vector-ivf")
            for i in top
        ]

    def probed_fraction(self) -> float:
        """Fraction of the corpus an average query actually scans."""
        if self._centroids is None and self._vectors is not None:
            self._build()
        return min(1.0, self.n_probe / max(1, self.n_lists))


def recall_at_k(approximate: Sequence[ScoredChunk], exact: Sequence[ScoredChunk]) -> float:
    """Overlap between approximate and exact results. The ANN quality metric.

    recall@10 of 0.9 means the approximate index found 9 of the 10 true
    nearest neighbours. Below ~0.95 users start noticing worse answers.
    """
    if not exact:
        return 1.0
    exact_ids = {s.chunk.chunk_id for s in exact}
    found = sum(1 for s in approximate if s.chunk.chunk_id in exact_ids)
    return found / len(exact_ids)


class ChromaVectorStore:
    """Chroma-backed store, for when you want persistence without a server.

    Optional dependency. The interface is identical to InMemoryVectorStore,
    which is the point of the Protocol: swapping storage should not touch
    retrieval, generation, or evaluation code.
    """

    def __init__(
        self, collection_name: str = "ragkit", persist_directory: Optional[str] = None
    ) -> None:
        try:
            import chromadb  # optional
        except ImportError as exc:  # pragma: no cover - needs the optional dep
            raise ImportError("pip install chromadb (see requirements-optional.txt)") from exc

        client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.EphemeralClient()
        )
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._by_id: dict[str, Chunk] = {}

    def __len__(self) -> int:
        return int(self._collection.count())

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "doc_id": c.doc_id, **c.metadata} for c in chunks],
        )
        for chunk in chunks:
            self._by_id[chunk.chunk_id] = chunk

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        *,
        where: Optional[MetadataFilter] = None,
    ) -> list[ScoredChunk]:
        # Over-fetch when filtering in Python, so post-filtering does not
        # starve the result set. A real deployment pushes the filter down into
        # Chroma's `where=` clause instead.
        fetch = k * 5 if where else k
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector).ravel().tolist()], n_results=fetch
        )
        out: list[ScoredChunk] = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            chunk = self._by_id.get(chunk_id)
            if chunk is None or (where is not None and not where(chunk)):
                continue
            out.append(ScoredChunk(chunk, 1.0 - float(distance), retriever="vector-chroma"))
            if len(out) >= k:
                break
        return out


def index_memory_estimate(n_vectors: int, dimensions: int) -> str:
    """Back-of-envelope index size. Useful in system-design interviews.

    >>> index_memory_estimate(1_000_000, 1536)
    '5.7 GB'
    """
    total_bytes = n_vectors * dimensions * 4  # float32
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total_bytes < 1024 or unit == "TB":
            return f"{total_bytes:.1f} {unit}"
        total_bytes /= 1024.0
    return f"{total_bytes:.1f} TB"
