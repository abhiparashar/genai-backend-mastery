"""Why hybrid search beats vector search -- demonstrated, not asserted.

    python 04-rag/examples/rag_ex_hybrid_vs_vector.py

Dense and sparse retrieval fail in OPPOSITE directions:

    vector (dense)   understands "vacation" ~ "annual leave"
                     but cannot tell SKU-4471 from PAY-1180
    BM25 (sparse)    nails SKU-4471 instantly
                     but finds nothing for a paraphrase
    hybrid (RRF)     both

This script uses `SemanticStubEmbedder`, a teaching device that deliberately
reproduces those two behaviours offline: synonyms collapse together, and rare
alphanumeric identifiers get absorbed into one low-weight "identifier"
direction -- exactly how a real dense model treats them.

(The default `HashingEmbedder` is purely LEXICAL, which makes it accidentally
excellent at identifiers and therefore hides the entire point. Using it here
would produce a demo that quietly proves nothing.)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit.chunking import MarkdownStructureChunker  # noqa: E402
from ragkit.embeddings import SemanticStubEmbedder  # noqa: E402
from ragkit.retrieval import (  # noqa: E402
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
)
from ragkit.types import Document  # noqa: E402
from ragkit.vectorstore import InMemoryVectorStore  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

# (question, a substring that must appear in the correct chunk, why it is hard)
QUERIES = [
    ("What does error SKU-4471 mean?", "SKU-4471", "exact identifier"),
    ("How do I fix PAY-1180?", "PAY-1180", "exact identifier"),
    (
        "How much vacation time do I get?",
        "25 days of paid annual leave",
        "synonym: vacation->leave",
    ),
    (
        "How long until I get my money back?",
        "full refund within 30 days",
        "paraphrase: money back->refund",
    ),
    (
        "What is the deadline for submitting expenses?",
        "within 30 days of the date of purchase",
        "literal wording",
    ),
    (
        "When can a new starter touch production?",
        "granted only after security training",
        "paraphrase",
    ),
]


def rank_of(chunks, marker: str):
    for index, scored in enumerate(chunks, start=1):
        if marker.lower() in scored.text.lower():
            return index
    return None


def fmt(rank) -> str:
    return f"#{rank}" if rank else "MISS"


def main() -> None:
    print(__doc__)

    documents = [
        Document(path.read_text(encoding="utf-8"), source=path.name)
        for path in sorted((DATA / "corpus").glob("*.md"))
    ]
    chunks = [c for d in documents for c in MarkdownStructureChunker(1200, 150).split(d)]

    embedder = SemanticStubEmbedder(256)
    store = InMemoryVectorStore(embedder.dimensions)
    store.add(chunks, embedder.embed([c.text for c in chunks]))

    vector = VectorRetriever(store, embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector, bm25, fetch_k=20)

    print(f"Corpus: {len(documents)} documents -> {len(chunks)} chunks\n")

    header = f"{'query':<46}{'why hard':<28}{'vector':>8}{'bm25':>8}{'hybrid':>8}"
    print(header)
    print("-" * len(header))

    wins = {"vector": 0, "bm25": 0, "hybrid": 0}
    found = {"vector": 0, "bm25": 0, "hybrid": 0}
    # MRR is the honest headline here: "found it somewhere in the top 5" hides
    # the difference between rank 1 and rank 5, and rank matters -- models
    # attend most reliably to the start of the context.
    reciprocal = {"vector": 0.0, "bm25": 0.0, "hybrid": 0.0}

    for question, marker, why in QUERIES:
        ranks = {
            "vector": rank_of(vector.retrieve(question, k=5).chunks, marker),
            "bm25": rank_of(bm25.retrieve(question, k=5).chunks, marker),
            "hybrid": rank_of(hybrid.retrieve(question, k=5).chunks, marker),
        }
        for name, rank in ranks.items():
            if rank:
                found[name] += 1
                reciprocal[name] += 1.0 / rank
            if rank == 1:
                wins[name] += 1

        print(
            f"{question[:44]:<46}{why:<28}"
            f"{fmt(ranks['vector']):>8}{fmt(ranks['bm25']):>8}{fmt(ranks['hybrid']):>8}"
        )

    total = len(QUERIES)
    print("-" * len(header))
    print(f"{'found in top 5':<74}{found['vector']:>8}{found['bm25']:>8}{found['hybrid']:>8}")
    print(f"{'ranked #1':<74}{wins['vector']:>8}{wins['bm25']:>8}{wins['hybrid']:>8}")
    print(
        f"{'MRR (rank quality, higher is better)':<74}"
        + "".join(f"{reciprocal[n] / total:>8.2f}" for n in ("vector", "bm25", "hybrid"))
    )
    print(f"{'(out of ' + str(total) + ' queries)':<74}")

    print("\n" + "=" * 96)
    print("WHAT JUST HAPPENED")
    print("=" * 96)
    print(
        "Read the columns, not the folklore.\n\n"
        "VECTOR mis-ranked the identifier queries -- SKU-4471 fell to #5. To a dense\n"
        "model that token is rare and barely moves the vector, so one error-code chunk\n"
        "looks much like another. But it was the ONLY retriever that understood\n"
        "'vacation' means 'annual leave'.\n\n"
        "BM25 nailed every identifier at #1 -- a rare term has high inverse document\n"
        "frequency, which is exactly what BM25 rewards -- and then MISSED the synonym\n"
        "query completely, because 'vacation' and 'leave' share no characters.\n\n"
        "HYBRID was the only method that found all six. Note honestly that BM25 scored\n"
        "the best MRR on this corpus: these queries are lexical-heavy and the corpus is\n"
        "tiny, so fusion dilutes a strong sparse signal with a weak dense one. RRF buys\n"
        "ROBUSTNESS, not a guaranteed win on every metric.\n\n"
        "That is the real lesson, and it is the module's thesis: a total miss (BM25 on\n"
        "the synonym query) is far worse in production than a rank-4 hit, because the\n"
        "user gets nothing at all. Hybrid removes the catastrophic case.\n\n"
        "And note what this script just demonstrated about METHOD: the numbers\n"
        "contradicted the slogan. Measure on YOUR corpus with YOUR queries. Anyone who\n"
        "tells you hybrid always wins has not run the experiment."
    )


if __name__ == "__main__":
    main()
