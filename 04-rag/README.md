# Module 04 — RAG (`ragkit`)

**Phase 4 of the roadmap · Weeks 21–30 · the longest phase, because RAG is most of the job market**

Retrieval-Augmented Generation built from scratch — no LangChain. You implement the recursive splitter, BM25, cosine search, RRF fusion and MMR yourself, so that when a framework misbehaves you know what it was *supposed* to do.

Everything runs offline. 64 tests, no API key, no network.

```bash
python 04-rag/examples/rag_ex_hybrid_vs_vector.py   # why hybrid exists
python 04-rag/examples/rag_ex_evaluate.py           # the scorecard workflow
pytest 04-rag/tests -q
```

---

## What RAG actually is

Give the model the right text, in the prompt, at the moment it answers.

```
INGEST   documents → chunk → embed → store (+ BM25 index)
QUERY    question → retrieve → rerank → fit to budget → prompt → answer
```

That's it. Everything else — hybrid search, reranking, HyDE, parent-child — is engineering to make *"the right text"* more reliably right.

### RAG vs fine-tuning

The question every interviewer asks. The distinction is **knowledge vs behaviour**:

| | RAG | Fine-tuning |
|---|---|---|
| Gives the model | **facts** it didn't have | **behaviour/format/tone** |
| Update cost | re-index one document | retrain |
| Freshness | immediate | frozen at training time |
| Citations | yes, verifiable | no |
| Per-request cost | higher (context tokens) | lower |
| "Answer in our house style" | poor fit | good fit |
| "What's our refund policy?" | good fit | poor fit |

Default to RAG. Reach for fine-tuning when you need consistent *form*, not new *facts*. They compose.

---

## The chunking tension

A chunk is simultaneously **the unit of matching** (must be focused to score well) and **the unit of context** (must be complete to answer from). Those pull in opposite directions — and that tension is the whole problem.

| Size | Failure mode |
|---|---|
| 128 tokens | precise retrieval, but the answer is split across chunks and the model sees fragments |
| 2000+ tokens | every chunk is about everything, embeddings blur toward the document average, retrieval returns "anything vaguely on topic" |

**Start at ~512 tokens with 10–15% overlap, split on structure, then measure.**

Bad splits are **silent**. Cut a table in half or separate a heading from its content and retrieval quietly degrades with no error anywhere. Chunking bugs don't crash; they just make your product mediocre.

Four strategies in `chunking.py`:

| Strategy | Use when |
|---|---|
| `FixedSizeChunker` | control/baseline, or genuinely unstructured text |
| `RecursiveCharacterChunker` | the sane default — paragraph → line → sentence → word |
| `MarkdownStructureChunker` | **usually the winner on docs.** A heading is an author-declared boundary, and the heading *path* disambiguates the chunk |
| `SemanticChunker` | long unstructured prose only. Costs one embedding per sentence and is *not* reliably better |

> `"Employees get 25 days"` is nearly useless in isolation. `"Handbook > Benefits > Annual Leave: employees get 25 days"` retrieves correctly *and* reads correctly. That prefix is one of the cheapest quality wins available.

---

## Why hybrid search, in one table

Dense and sparse fail in **opposite** directions:

| | good at | blind to |
|---|---|---|
| **Vector** | paraphrase, synonyms, intent | identifiers, error codes, rare terms |
| **BM25** | `SKU-4471`, names, acronyms | "vacation" ≈ "annual leave" |

Real output from `rag_ex_hybrid_vs_vector.py`:

```
query                              why hard                     vector  bm25  hybrid
What does error SKU-4471 mean?     exact identifier                 #5    #1      #3
How much vacation time do I get?   synonym: vacation->leave         #1  MISS      #4
When can a new starter touch prod? paraphrase                       #3    #1      #2
------------------------------------------------------------------------------------
found in top 5                                                       6     5       6
MRR                                                               0.59  0.83    0.60
```

**Read that honestly: BM25 wins MRR here.** These queries are lexical-heavy on a tiny corpus, so fusion dilutes a strong sparse signal with a weak dense one. What hybrid actually buys is **robustness** — it was the only method to find all six, while BM25 *missed the synonym query entirely*.

A total miss is far worse in production than a rank-4 hit: the user gets nothing. Hybrid removes the catastrophic case.

And note the method lesson: **the numbers contradicted the slogan.** Measure on your corpus, with your queries.

---

## Evaluation: the part that separates engineering from vibes

Every RAG demo works. The question is whether yours works on the 200 questions your users will actually ask — and whether the chunk-size change you just made helped or hurt.

**Evaluate the two halves separately.** This is the single most useful diagnostic idea in the module:

```
retrieval failed   → the right chunk was never fetched
                     fix: chunking, embeddings, hybrid, reranking
generation failed  → the right chunk WAS fetched and the model still got it wrong
                     fix: the prompt, the model, context ordering
```

Check `hit_rate` first. **At 0.6 retrieval, no prompt engineering will save you**, and every hour spent on the prompt is wasted.

| Metric | Answers |
|---|---|
| `hit_rate` | did any relevant chunk make top-k? *(blunt, vital, fix first)* |
| `precision@k` | how much of what we sent was noise? |
| `MRR` / `NDCG@k` | was it ranked *near the top*? Position matters |
| **`faithfulness`** | **is every claim supported by the context? ← the hallucination metric** |
| `answer_relevance` | did it address the question? |
| `correctness` | token F1 vs expected |
| `refusal_accuracy` | did it say "I don't know" when it should? |

The golden set (`data/golden_qa.json`, 16 questions) deliberately spans failure modes — exact-identifier, synonym, multi-fact, inference, distractor-risk, and **two unanswerable questions**. A golden set with no unanswerable questions cannot detect a system that never admits ignorance.

Labels are **substrings, not chunk ids**, because chunk ids change whenever you change chunk size — which would invalidate your dataset exactly when you need it.

---

## The three production guardrails

1. **Verifiable citations.** `citation_check()` catches a model emitting `[7]` when 5 sources were sent. Users click through, find nothing, and stop trusting *every* citation — including the correct ones.
2. **A context budget.** Chunks are dropped, never truncated mid-sentence — a half-sentence is worse than none, because the model confidently completes it.
3. **A refusal path that costs zero tokens.** When the best chunk scores below threshold, answer "I don't know". A system that admits ignorance is trusted; one that guesses fluently is abandoned after the first bad answer.

> **More context is not free.** Cost and latency scale linearly with input tokens, and accuracy *degrades* — models attend less reliably to the middle of a long context ("lost in the middle"). Five good chunks beat twenty mediocre ones on every axis.

---

## Do you need a vector database?

Probably not yet.

| Scale | Use |
|---|---|
| < 100k chunks | **numpy.** Exact search in milliseconds. `InMemoryVectorStore` |
| persistence needed | **pgvector.** You already run Postgres |
| > 1M vectors, or ANN/filtering at scale | a dedicated vector DB |

`index_memory_estimate(1_000_000, 1536)` → **5.7 GB**. Dimension is a cost decision as much as a quality one.

`IVFIndex` implements FAISS's clustering idea in ~40 lines so the speed/recall tradeoff is *measured*, not claimed:

| `n_probe` | corpus scanned | recall@10 |
|---|---|---|
| 1 | ~6% | 0.55 |
| 3 | ~19% | 0.85 |
| 8 | ~50% | 0.99 |
| 16 | 100% | 1.00 |

### The filtering trap

**Pre-filter, don't post-filter.** Post-filtering searches globally then discards non-matching results — so a tenant-A query can legitimately return *zero* rows because all 10 nearest vectors belonged to tenant B. That's a correctness bug, and in multi-tenant systems it's a **data-isolation** bug. `tenant_filter()` plus a test asserting no cross-tenant leakage.

---

## Troubleshooting: why your RAG is bad

| Symptom | Likely cause | Fix |
|---|---|---|
| Right doc never retrieved | chunking split the fact from its context | structure-aware chunking; add heading paths |
| Fails on error codes / IDs / names | pure vector search | add BM25 → hybrid |
| Retrieves the topic but the wrong section | chunks too large | reduce chunk size; parent-child |
| Top-5 are near-duplicates | redundant corpus | MMR |
| Relevant chunk retrieved but ranked 8th | no reranking | cross-encoder rerank |
| Answer ignores provided context | context too long, or buried | tighten the budget; reorder |
| Confident nonsense | no refusal path | raise `min_score`; check `faithfulness` |
| Ingest cost keeps climbing | re-embedding unchanged docs | content-hash skip (built in) |
| Quality collapsed after a change | mixed embedding models | dimension check; re-index everything |

---

## Files

| File | What's in it |
|---|---|
| `types.py` | `Document`, `Chunk`, `ScoredChunk`, `Answer`, offline `FakeProvider` |
| `chunking.py` | Four strategies + `chunk_stats` |
| `embeddings.py` | `HashingEmbedder` (offline), `SemanticStubEmbedder` (teaching device), sentence-transformers + OpenAI, caching |
| `vectorstore.py` | Exact search, `IVFIndex`, Chroma, pre-filtering, memory estimates |
| `retrieval.py` | BM25, vector, hybrid RRF, MMR, rerankers, multi-query, HyDE, parent-child |
| `pipeline.py` | End-to-end wiring, citations, budget, refusal |
| `evaluation.py` | Golden sets, retrieval + generation metrics, scorecards, `compare()` |
| `data/` | 6-doc corpus + 16-question golden set |

---

## Java ↔ Python

| Java world | Here |
|---|---|
| Lucene / Elasticsearch BM25 | `BM25Retriever` — same algorithm, same `k1=1.5, b=0.75` |
| Postgres + pgvector | `InMemoryVectorStore` → swap to pgvector in production |
| Caffeine | `CachingEmbedder` |
| An interface with two impls | `Protocol` — `Embedder`, `VectorStore`, `Retriever`, `Chunker` |
| JUnit golden-file tests | `data/golden_qa.json` + `evaluate()` |

---

## Exit criteria

1. You can draw the ingest and query paths from memory.
2. You can explain when BM25 beats embeddings, **with an example**.
3. You can produce a scorecard, change chunk size, and explain the movement.
4. You can name the failure mode that pre-filtering prevents.
5. You can argue RAG vs fine-tuning for a specific feature.

**Next:** [Module 05 — Agents](../05-agents/) — when retrieval isn't enough and the model needs to *act*.
