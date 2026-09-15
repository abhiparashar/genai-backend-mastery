"""Measure RAG instead of guessing at it.

    python 04-rag/examples/rag_ex_evaluate.py

Runs the golden set against several configurations and prints a scorecard for
each, then a side-by-side comparison. Entirely offline.

This is the workflow the whole module exists to teach: change ONE variable,
re-run, read the numbers. Not "that feels better".

Note the embedder: `HashingEmbedder` captures lexical overlap, not semantics.
Absolute scores here are therefore lower than you would see with a real
embedding model. The COMPARISONS are still meaningful, which is the point --
and swapping in SentenceTransformerEmbedder changes one line.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragkit.chunking import (  # noqa: E402
    FixedSizeChunker,
    MarkdownStructureChunker,
    RecursiveCharacterChunker,
)
from ragkit.evaluation import compare, evaluate, load_golden  # noqa: E402
from ragkit.pipeline import RAGPipeline  # noqa: E402
from ragkit.types import Document, FakeProvider  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def load_corpus() -> list[Document]:
    return [
        Document(path.read_text(encoding="utf-8"), source=path.name)
        for path in sorted((DATA / "corpus").glob("*.md"))
    ]


def grounded_answer(messages) -> str:
    """A deterministic stand-in for a real model.

    It answers strictly FROM the retrieved context: it echoes the sentence
    that best overlaps the question. That is deliberate -- it means the
    scorecard measures RETRIEVAL quality without a real model's noise on top.
    Generation metrics here are therefore an upper bound given the context
    that was supplied.
    """
    prompt = messages[-1].content
    if "Sources:" not in prompt or "Question:" not in prompt:
        return "I don't know based on the provided sources."

    sources_block = prompt.split("Sources:", 1)[1].split("Question:", 1)[0]
    question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip().lower()
    question_words = {w for w in question.replace("?", "").split() if len(w) > 3}

    best_sentence, best_overlap, best_index = "", 0, 1
    for block in sources_block.strip().split("\n\n"):
        index = 1
        if block.startswith("["):
            try:
                index = int(block[1 : block.index("]")])
            except (ValueError, IndexError):
                index = 1
        for sentence in block.replace("\n", " ").split(". "):
            words = {w.lower().strip(".,") for w in sentence.split() if len(w) > 3}
            overlap = len(question_words & words)
            if overlap > best_overlap:
                best_sentence, best_overlap, best_index = sentence.strip(), overlap, index

    if best_overlap == 0:
        return "I don't know based on the provided sources."
    return f"{best_sentence}. [{best_index}]"


def build(**kwargs) -> RAGPipeline:
    pipeline = RAGPipeline(provider=FakeProvider(answer_fn=grounded_answer), **kwargs)
    pipeline.ingest(load_corpus())
    return pipeline


def main() -> None:
    print(__doc__)
    golden = load_golden(str(DATA / "golden_qa.json"))
    print(
        f"Golden set: {len(golden)} questions "
        f"({sum(1 for e in golden if e.should_refuse)} deliberately unanswerable)\n"
    )

    configs = {
        "markdown+hybrid": build(chunker=MarkdownStructureChunker(1200, 150), use_hybrid=True),
        "markdown+vector": build(chunker=MarkdownStructureChunker(1200, 150), use_hybrid=False),
        "recursive+hybrid": build(chunker=RecursiveCharacterChunker(1200, 150), use_hybrid=True),
        "fixed+hybrid": build(chunker=FixedSizeChunker(1200, 150), use_hybrid=True),
        "tiny-chunks": build(chunker=RecursiveCharacterChunker(200, 20), use_hybrid=True),
        "huge-chunks": build(chunker=RecursiveCharacterChunker(6000, 0), use_hybrid=True),
    }

    cards = {}
    for name, pipeline in configs.items():
        cards[name] = evaluate(pipeline, golden)
        print(f"  {name:<18} chunks={len(pipeline.chunks):<4}")

    print("\n" + "=" * 100)
    print("SCORECARD COMPARISON")
    print("=" * 100)
    print(compare(cards))

    print("\n" + "=" * 100)
    print("DETAIL: markdown+hybrid")
    print("=" * 100)
    print(cards["markdown+hybrid"].report())

    print("\n" + "=" * 100)
    print("WHERE RETRIEVAL MISSED (read these, not the average)")
    print("=" * 100)
    misses = cards["markdown+hybrid"].failures("hit_rate", below=1.0)
    if misses:
        for result in misses:
            print(f"  MISS  {result.question}")
    else:
        print("  none -- every answerable question retrieved a relevant chunk")

    print("\n" + "=" * 100)
    print("THE EXACT-IDENTIFIER CASE: why hybrid exists")
    print("=" * 100)
    identifier_questions = [e for e in golden if e.metadata.get("type") == "exact-identifier"]
    for example in identifier_questions:
        hybrid = configs["markdown+hybrid"].retrieve(example.question)
        vector = configs["markdown+vector"].retrieve(example.question)
        marker = example.relevant_chunk_texts[0]
        hybrid_rank = next(
            (i for i, c in enumerate(hybrid.chunks, 1) if marker.lower() in c.text.lower()), None
        )
        vector_rank = next(
            (i for i, c in enumerate(vector.chunks, 1) if marker.lower() in c.text.lower()), None
        )
        print(f"  {example.question}")
        print(
            f"      hybrid rank: {hybrid_rank or 'NOT FOUND'}   vector-only rank: {vector_rank or 'NOT FOUND'}"
        )

    print("\n" + "=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print(
        "1. hit_rate first. If retrieval misses, no prompt change will help.\n"
        "2. Compare ONE variable at a time. tiny-chunks vs huge-chunks shows\n"
        "   the precision/context tradeoff directly in the numbers.\n"
        "3. faithfulness is your hallucination alarm. Watch it when you change\n"
        "   the prompt or the model.\n"
        "4. Commit this scorecard. When it regresses, you know which change did it."
    )


if __name__ == "__main__":
    main()
