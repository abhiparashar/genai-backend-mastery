"""Retrieval, storage, fusion, and the end-to-end pipeline.

Scores are asserted against hand-computed values where the maths is
checkable, and against ORDERING where it is not. Never against a magic
constant someone read off a previous run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from ragkit.embeddings import (
    CachingEmbedder,
    HashingEmbedder,
    SemanticStubEmbedder,
    cosine_similarity,
    normalize,
)
from ragkit.evaluation import (
    GoldenExample,
    answer_correctness,
    evaluate,
    faithfulness,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
)
from ragkit.pipeline import build_pipeline, citation_check, fit_to_budget, format_context
from ragkit.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LLMReranker,
    ParentChildRetriever,
    VectorRetriever,
    hyde,
    maximal_marginal_relevance,
    multi_query,
    reciprocal_rank_fusion,
)
from ragkit.types import Answer, Chunk, Document, FakeProvider, ScoredChunk
from ragkit.vectorstore import (
    InMemoryVectorStore,
    IVFIndex,
    index_memory_estimate,
    recall_at_k,
    tenant_filter,
)

CORPUS = [
    "To reset your password visit account settings and click forgot login.",
    "Error SKU-4471 indicates the warehouse inventory sync failed for that item.",
    "Annual leave is 25 days per year for all permanent employees.",
    "If the application keeps crashing, clear the local cache and restart.",
    "Expenses must be submitted within 30 days of purchase to be reimbursed.",
    "Error PAY-9902 means the payment gateway rejected the transaction.",
]


@pytest.fixture()
def chunks():
    return [Chunk(t, source=f"doc{i}.md", chunk_id=f"c{i}") for i, t in enumerate(CORPUS)]


@pytest.fixture()
def embedder():
    return HashingEmbedder(dimensions=128)


@pytest.fixture()
def store(chunks, embedder):
    store = InMemoryVectorStore(128)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_cosine_similarity_matches_known_values():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_normalize_produces_unit_vectors_and_survives_zeros():
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0)
    # A zero row must not produce NaN -- empty strings happen.
    assert np.isfinite(matrix[1]).all()


def test_related_text_scores_above_unrelated(embedder):
    vectors = embedder.embed(["the cat sat on the mat", "a cat sat on a mat", "bond yields rose"])
    assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(vectors[0], vectors[2])


def test_embeddings_are_deterministic(embedder):
    assert np.array_equal(embedder.embed(["stable"]), embedder.embed(["stable"]))


def test_caching_embedder_returns_identical_vectors_to_the_inner_one():
    inner = HashingEmbedder(64)
    cached = CachingEmbedder(HashingEmbedder(64))
    assert np.allclose(cached.embed(["hello"]), inner.embed(["hello"]))
    cached.embed(["hello"])
    assert cached.hits == 1


def test_semantic_stub_models_dense_behaviour():
    stub = SemanticStubEmbedder(256)
    vectors = stub.embed(["annual leave policy", "vacation holiday time off"])
    # Synonyms collapse -- the behaviour a lexical matcher cannot reproduce.
    assert cosine_similarity(vectors[0], vectors[1]) > 0.5
    # And identifiers blur, which is why vector search fails on them.
    assert stub._canonical("4471")[0] == "generic_identifier"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def test_search_returns_the_most_similar_chunk_first(store, embedder):
    hits = store.search(embedder.embed(["password reset login"])[0], k=3)
    assert hits[0].chunk.chunk_id == "c0"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_pre_filtering_returns_a_full_result_set(chunks, embedder):
    # Post-filtering would search globally then discard, and could return
    # nothing for a tenant whose chunks all rank below the global top-k.
    tenanted = [
        Chunk(
            c.text,
            source=c.source,
            chunk_id=c.chunk_id,
            metadata={"tenant": "acme" if i < 2 else "globex"},
        )
        for i, c in enumerate(chunks)
    ]
    store = InMemoryVectorStore(128)
    store.add(tenanted, embedder.embed([c.text for c in tenanted]))

    hits = store.search(embedder.embed(["annual leave days"])[0], k=2, where=tenant_filter("acme"))
    assert len(hits) == 2
    assert all(h.chunk.metadata["tenant"] == "acme" for h in hits)


def test_tenant_isolation_never_leaks_another_tenants_text(chunks, embedder):
    # This failure mode is a data breach, not a bug.
    tenanted = [
        Chunk(
            c.text,
            source=c.source,
            chunk_id=c.chunk_id,
            metadata={"tenant": "acme" if i % 2 == 0 else "globex"},
        )
        for i, c in enumerate(chunks)
    ]
    store = InMemoryVectorStore(128)
    store.add(tenanted, embedder.embed([c.text for c in tenanted]))

    for query in ("password", "leave", "error code", "expenses"):
        hits = store.search(embedder.embed([query])[0], k=10, where=tenant_filter("acme"))
        assert all(h.chunk.metadata["tenant"] == "acme" for h in hits)


def test_dimension_mismatch_is_rejected_loudly(store):
    # Mixing embedding models is otherwise a silent quality collapse.
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add([Chunk("x")], np.zeros((1, 999), dtype=np.float32))


def test_delete_removes_chunks_and_their_vectors(store):
    before = len(store)
    removed = store.delete(lambda c: c.chunk_id == "c0")
    assert removed == 1
    assert len(store) == before - 1
    assert store.vectors.shape[0] == before - 1


def test_empty_store_returns_no_results(embedder):
    assert InMemoryVectorStore(128).search(embedder.embed(["anything"])[0]) == []


def test_ann_recall_improves_as_more_clusters_are_probed(embedder):
    rng = np.random.default_rng(7)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    many = [Chunk(" ".join(rng.choice(words, 6)), chunk_id=f"m{i}") for i in range(300)]
    vectors = embedder.embed([c.text for c in many])

    exact = InMemoryVectorStore(128)
    exact.add(many, vectors)

    def mean_recall(n_probe: int) -> float:
        index = IVFIndex(128, n_lists=16, n_probe=n_probe)
        index.add(many, vectors)
        scores = [
            recall_at_k(index.search(vectors[i], k=10), exact.search(vectors[i], k=10))
            for i in range(0, 60, 6)
        ]
        return sum(scores) / len(scores)

    low, high = mean_recall(1), mean_recall(16)
    assert low < high
    assert high == pytest.approx(1.0), "probing every cluster must equal exact search"


def test_index_memory_estimate_is_right():
    # 1M x 1536 x 4 bytes = 5.72 GiB. Worth knowing in a design interview.
    assert index_memory_estimate(1_000_000, 1536) == "5.7 GB"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_ranks_the_exact_identifier_first(chunks):
    hits = BM25Retriever(chunks).retrieve("SKU-4471", k=3)
    assert hits.chunks[0].chunk.chunk_id == "c1"


def test_rare_terms_outweigh_common_ones(chunks):
    bm25 = BM25Retriever(chunks)
    assert bm25.idf("sku") > bm25.idf("the")


def test_term_frequency_saturates(chunks):
    # The fix for keyword stuffing: 20 occurrences must not score 20x.
    bm25 = BM25Retriever([Chunk("spam", chunk_id="a"), Chunk("spam " * 20, chunk_id="b")])
    single = bm25.score(["spam"], 0)
    many = bm25.score(["spam"], 1)
    assert many > single
    assert many < single * 3


def test_bm25_returns_nothing_for_unmatched_terms(chunks):
    assert BM25Retriever(chunks).retrieve("zzzz nonexistent qqqq", k=5).chunks == []


# ---------------------------------------------------------------------------
# Fusion and diversity
# ---------------------------------------------------------------------------


def test_rrf_uses_rank_not_incommensurable_scores():
    # BM25 (0..20) and cosine (-1..1) are not comparable; rank always is.
    big = [ScoredChunk(Chunk("a", chunk_id="A"), 19.0)]
    small = [ScoredChunk(Chunk("b", chunk_id="B"), 0.4)]
    fused = reciprocal_rank_fusion([big, small])
    assert {s.chunk.chunk_id for s in fused} == {"A", "B"}
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_promotes_a_document_both_retrievers_like():
    agreed = Chunk("agreed", chunk_id="AGREE")
    first = [ScoredChunk(Chunk("x", chunk_id="X"), 1.0), ScoredChunk(agreed, 0.9)]
    second = [ScoredChunk(Chunk("y", chunk_id="Y"), 1.0), ScoredChunk(agreed, 0.9)]
    assert reciprocal_rank_fusion([first, second])[0].chunk.chunk_id == "AGREE"


def test_hybrid_finds_what_bm25_alone_misses(chunks, store, embedder):
    hybrid = HybridRetriever(VectorRetriever(store, embedder), BM25Retriever(chunks), fetch_k=10)
    # A term absent from the corpus: BM25 scores nothing, hybrid still returns
    # candidates from the dense side rather than an empty page.
    assert len(hybrid.retrieve("zzzz nonexistent qqqq", k=3).chunks) > 0


def test_mmr_reduces_redundancy(embedder):
    texts = ["cats are great pets"] * 4 + ["bond yields fell sharply"]
    duplicates = [Chunk(t, chunk_id=f"d{i}") for i, t in enumerate(texts)]
    vectors = embedder.embed(texts)
    candidates = [ScoredChunk(c, 1.0) for c in duplicates]
    query = embedder.embed(["cats"])[0]

    relevance_only = maximal_marginal_relevance(query, candidates, vectors, k=3, lambda_param=1.0)
    diverse = maximal_marginal_relevance(query, candidates, vectors, k=3, lambda_param=0.3)

    assert [s.chunk.chunk_id for s in relevance_only] == ["d0", "d1", "d2"]
    assert "d4" in [s.chunk.chunk_id for s in diverse], "diversity must surface the outlier"


# ---------------------------------------------------------------------------
# Rerank and query transforms
# ---------------------------------------------------------------------------


def test_llm_reranker_reorders_by_judged_score(chunks):
    provider = FakeProvider(
        answer_fn=lambda m: "9" if "SKU-4471" in m[-1].content.split("Passage:")[1] else "1"
    )
    ranked = LLMReranker(complete=provider.complete).rerank(
        "what is SKU-4471", [ScoredChunk(c, 0.5) for c in chunks], k=2
    )
    assert ranked[0].chunk.chunk_id == "c1"
    assert provider.calls == len(chunks), "one judgement per candidate -- the documented cost"


def test_multi_query_always_keeps_the_original():
    provider = FakeProvider(answer_fn=lambda m: "change password\nforgot credentials")
    variants = multi_query(provider.complete, "reset password", n=2)
    assert variants[0] == "reset password"
    assert len(variants) > 1


def test_hyde_returns_a_hypothetical_passage():
    provider = FakeProvider(answer_fn=lambda m: "Refunds are accepted within 30 days.")
    assert "30 days" in hyde(provider.complete, "what is the refund window?")


def test_parent_child_returns_parents_for_matched_children(embedder):
    children = [
        Chunk("25 days", chunk_id="k1", metadata={"parent_id": "P1"}),
        Chunk("30 days expenses", chunk_id="k2", metadata={"parent_id": "P2"}),
    ]
    store = InMemoryVectorStore(128)
    store.add(children, embedder.embed([c.text for c in children]))
    parents = {
        "P1": Chunk("FULL leave section", chunk_id="P1"),
        "P2": Chunk("FULL expenses", chunk_id="P2"),
    }

    results = ParentChildRetriever(VectorRetriever(store, embedder), parents).retrieve(
        "25 days", k=2
    )
    assert any("FULL" in s.text for s in results.chunks)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


DOCS = [
    Document(
        "# Leave\n\nAnnual leave is 25 days per year for permanent employees.", source="handbook.md"
    ),
    Document(
        "# Expenses\n\nExpenses must be submitted within 30 days of purchase.", source="expenses.md"
    ),
    Document(
        "# Errors\n\nError SKU-4471 means the warehouse inventory sync failed.", source="errors.md"
    ),
]


def test_ingest_skips_unchanged_documents():
    pipeline = build_pipeline(FakeProvider(responses=["ok"]))
    first = pipeline.ingest(DOCS)
    second = pipeline.ingest(DOCS)

    assert first["chunks"] > 0
    assert second["chunks"] == 0
    assert second["skipped"] == len(DOCS), "content hash must prevent re-embedding"


def test_changed_document_is_reindexed():
    pipeline = build_pipeline(FakeProvider(responses=["ok"]))
    pipeline.ingest(DOCS)
    updated = [Document("# Leave\n\nAnnual leave is now 30 days.", source="handbook.md")]
    assert pipeline.ingest(updated)["documents"] == 1


def test_pipeline_answers_from_retrieved_context():
    pipeline = build_pipeline(FakeProvider(answer_fn=lambda m: "Annual leave is 25 days [1]."))
    pipeline.ingest(DOCS)
    answer = pipeline.answer("how many annual leave days?")

    assert not answer.refused
    assert "25 days" in answer.text
    assert answer.chunks, "an answer must carry the evidence it used"


def test_weak_retrieval_refuses_without_calling_the_model():
    pipeline = build_pipeline(FakeProvider(answer_fn=lambda m: "a confident guess"), min_score=0.95)
    pipeline.ingest(DOCS)
    answer = pipeline.answer("who won the 1998 world cup?")

    assert answer.refused
    assert "don't know" in answer.text
    assert pipeline.provider.calls == 0, "refusing must cost zero tokens"


def test_refusal_threshold_is_comparable_across_retrievers():
    # Regression: the gate once compared min_score against RRF scores (~0.016),
    # which refused every query under hybrid search.
    for hybrid in (True, False):
        pipeline = build_pipeline(
            FakeProvider(answer_fn=lambda m: "25 days [1]"), min_score=0.1, use_hybrid=hybrid
        )
        pipeline.ingest(DOCS)
        assert not pipeline.answer("annual leave days").refused, f"use_hybrid={hybrid}"


def test_context_budget_is_never_exceeded():
    oversized = [ScoredChunk(Chunk("x" * 4000, chunk_id=f"b{i}"), 1.0 - i * 0.1) for i in range(10)]
    kept = fit_to_budget(oversized, max_tokens=2000)
    assert sum(len(c.text) for c in kept) // 4 <= 2000
    assert len(kept) < len(oversized)


def test_context_is_numbered_for_citation():
    rendered = format_context(
        [
            ScoredChunk(Chunk("first", source="a.md"), 1.0),
            ScoredChunk(Chunk("second", source="b.md"), 0.5),
        ]
    )
    assert "[1]" in rendered and "[2]" in rendered
    assert rendered.index("[1]") < rendered.index("[2]")


def test_citation_check_flags_a_fabricated_source_number():
    answer = Answer("as shown in [7]", chunks=[ScoredChunk(Chunk("only one"), 1.0)])
    report = citation_check(answer)
    assert report["valid"] is False
    assert report["dangling"] == [7]


def test_citation_check_accepts_valid_numbering():
    answer = Answer(
        "per [1] and [2]", chunks=[ScoredChunk(Chunk("a"), 1.0), ScoredChunk(Chunk("b"), 0.9)]
    )
    assert citation_check(answer)["valid"] is True


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_retrieval_metrics_match_hand_computed_values():
    hits = [
        ScoredChunk(Chunk("irrelevant"), 1.0),
        ScoredChunk(Chunk("the answer is 25 days"), 0.9),
        ScoredChunk(Chunk("also irrelevant"), 0.8),
    ]
    markers = ["25 days"]

    assert hit_rate(hits, markers) == 1.0
    assert precision_at_k(hits, markers) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(hits, markers) == 0.5  # relevant chunk at rank 2
    # NDCG: single hit at rank 2 -> (1/log2(3)) / (1/log2(2))
    assert ndcg_at_k(hits, markers) == pytest.approx((1 / math.log2(3)) / (1 / math.log2(2)))


def test_metrics_are_zero_when_nothing_relevant_is_retrieved():
    hits = [ScoredChunk(Chunk("nothing useful"), 1.0)]
    assert hit_rate(hits, ["missing"]) == 0.0
    assert mean_reciprocal_rank(hits, ["missing"]) == 0.0


def test_faithfulness_separates_grounded_from_invented_answers():
    context = [ScoredChunk(Chunk("Annual leave is 25 days per year for permanent employees."), 1.0)]
    grounded = Answer("Annual leave is 25 days per year.", chunks=context)
    invented = Answer("The CEO is Bob Smith and revenue tripled in Belgium.", chunks=context)

    assert faithfulness(grounded) > 0.8
    assert faithfulness(invented) < 0.3


def test_correctness_uses_f1_not_exact_match():
    # "25 days" and the fuller sentence are both correct answers.
    assert answer_correctness(Answer("Employees get 25 days of annual leave"), "25 days") > 0.0
    assert answer_correctness(Answer("completely unrelated text"), "25 days") == 0.0


def test_evaluate_scores_a_pipeline_over_a_golden_set():
    pipeline = build_pipeline(
        FakeProvider(answer_fn=lambda m: "Annual leave is 25 days [1]."), min_score=0.1
    )
    pipeline.ingest(DOCS)
    golden = [
        GoldenExample("how many annual leave days?", "25 days", ["25 days"]),
        GoldenExample("what is the expense deadline?", "30 days", ["30 days"]),
    ]
    card = evaluate(pipeline, golden)

    assert len(card.results) == 2
    summary = card.summary()
    assert 0.0 <= summary["hit_rate"] <= 1.0
    assert "hit_rate" in card.report()
