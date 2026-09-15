"""Chunking correctness.

The invariant that matters most: **chunking must not lose content.** A silent
drop is the worst RAG bug there is -- retrieval simply never finds the answer,
and nothing anywhere reports an error.
"""

from __future__ import annotations

import pytest
from ragkit.chunking import (
    FixedSizeChunker,
    MarkdownStructureChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    chunk_stats,
)
from ragkit.embeddings import HashingEmbedder
from ragkit.types import Document

PROSE = "The quick brown fox jumps over the lazy dog. " * 40

MARKDOWN = """# Handbook

Intro paragraph.

## Benefits

### Annual Leave

Employees get 25 days of paid leave.

### Health

Private medical cover from day one.

## Expenses

Submit within 30 days.
"""


@pytest.mark.parametrize("length", [0, 1, 50, 99, 100, 101, 250, 1000])
def test_fixed_chunker_never_loses_characters(length):
    # Repetitive input is the adversarial case: a naive dedup that compares
    # chunk TEXT will drop a trailing window that merely looks like a
    # substring of the previous one.
    document = Document("x" * length)
    chunks = FixedSizeChunker(chunk_size=100, overlap=10).split(document)
    if length == 0:
        assert chunks == []
        return
    assert max(c.end for c in chunks) >= length


def test_fixed_chunker_rejects_non_advancing_window():
    # overlap >= chunk_size means the window never moves: an infinite loop.
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="chunk_size"):
        FixedSizeChunker(chunk_size=0)


def test_recursive_chunker_prefers_the_strongest_separator():
    # Two paragraphs that each fit: it must split on the blank line rather
    # than slicing mid-paragraph.
    document = Document("A" * 50 + "\n\n" + "B" * 50)
    chunks = RecursiveCharacterChunker(chunk_size=60, overlap=0).split(document)
    assert len(chunks) == 2
    assert set(chunks[0].text) == {"A"}
    assert set(chunks[1].text) == {"B"}


def test_recursive_chunker_preserves_all_words():
    chunks = RecursiveCharacterChunker(chunk_size=200, overlap=20).split(Document(PROSE))
    joined = " ".join(c.text for c in chunks)
    for word in ("quick", "brown", "fox", "lazy", "dog"):
        assert word in joined
    assert len(chunks) > 1


def test_recursive_chunker_respects_the_size_budget():
    chunks = RecursiveCharacterChunker(chunk_size=200, overlap=20).split(Document(PROSE))
    # Overlap is prepended, so a chunk may exceed chunk_size by up to overlap.
    assert all(len(c) <= 200 + 20 for c in chunks)


def test_overlap_carries_context_across_the_boundary():
    without = RecursiveCharacterChunker(chunk_size=200, overlap=0).split(Document(PROSE))
    with_overlap = RecursiveCharacterChunker(chunk_size=200, overlap=40).split(Document(PROSE))
    # Overlap duplicates text, so total characters must grow.
    assert sum(len(c) for c in with_overlap) > sum(len(c) for c in without)


def test_markdown_chunker_records_the_heading_path():
    chunks = MarkdownStructureChunker(chunk_size=400).split(Document(MARKDOWN))
    paths = [c.heading_path for c in chunks]
    assert "Handbook > Benefits > Annual Leave" in paths
    assert "Handbook > Expenses" in paths


def test_markdown_chunker_keeps_a_section_intact():
    chunks = MarkdownStructureChunker(chunk_size=400).split(Document(MARKDOWN))
    leave = next(c for c in chunks if "Annual Leave" in c.heading_path)
    # The whole point: the fact and its heading arrive together.
    assert "25 days" in leave.text
    assert "Annual Leave" in leave.text


def test_markdown_chunker_keeps_content_before_the_first_heading():
    chunks = MarkdownStructureChunker(chunk_size=400).split(Document(MARKDOWN))
    assert any("Intro paragraph" in c.text for c in chunks)


def test_markdown_chunker_falls_back_when_there_are_no_headings():
    chunks = MarkdownStructureChunker(chunk_size=200).split(Document(PROSE))
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_semantic_chunker_splits_at_the_topic_change():
    def embed(texts):
        # Deterministic stand-in: two clearly separated topics.
        return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]

    document = Document(
        "The cat sat. The cat purred. The cat slept. Stock markets fell. Bonds rallied. Rates rose."
    )
    chunks = SemanticChunker(embed_fn=embed, chunk_size=500).split(document)

    assert len(chunks) == 2
    assert "cat" in chunks[0].text and "cat" not in chunks[1].text
    # And it must not silently drop the tail.
    joined = " ".join(c.text for c in chunks)
    assert "Rates rose" in joined and "cat slept" in joined


def test_chunks_carry_provenance_back_to_the_document():
    document = Document(MARKDOWN, source="handbook.md")
    chunks = MarkdownStructureChunker(chunk_size=400).split(document)
    assert all(c.source == "handbook.md" for c in chunks)
    assert all(c.doc_id == document.doc_id for c in chunks)
    # Ordinals must be dense and ordered, or "chunk 3 of 7" is meaningless.
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_stable_across_runs():
    # Unstable ids would break incremental indexing and dedup.
    first = MarkdownStructureChunker().split(Document(MARKDOWN, source="h.md"))
    second = MarkdownStructureChunker().split(Document(MARKDOWN, source="h.md"))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_stats_describe_the_distribution():
    chunks = RecursiveCharacterChunker(chunk_size=200, overlap=20).split(Document(PROSE))
    stats = chunk_stats(chunks)
    assert stats["count"] == len(chunks)
    assert stats["min"] <= stats["p50"] <= stats["max"]


def test_empty_document_produces_no_chunks():
    for chunker in (
        FixedSizeChunker(),
        RecursiveCharacterChunker(),
        MarkdownStructureChunker(),
    ):
        assert chunker.split(Document("")) == []
        assert chunker.split(Document("   \n\n  ")) == []


def test_document_hash_tracks_content_not_path():
    # This is what makes incremental re-indexing possible.
    a = Document("same text", source="a.md")
    b = Document("same text", source="b.md")
    c = Document("different", source="a.md")
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_embedder_and_chunker_compose_over_a_real_document():
    chunks = MarkdownStructureChunker(chunk_size=400).split(Document(MARKDOWN))
    vectors = HashingEmbedder(64).embed([c.text for c in chunks])
    assert vectors.shape == (len(chunks), 64)
