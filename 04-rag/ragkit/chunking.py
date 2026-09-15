"""Chunking strategies -- the highest-leverage decision in RAG.

WHY CHUNKING DECIDES WHETHER YOUR RAG WORKS

Retrieval returns chunks. So a chunk is simultaneously:

- the unit of MATCHING  -- it must be focused enough to score well on a query
- the unit of CONTEXT   -- it must be complete enough to answer from

Those pull in opposite directions, and that tension is the whole problem.

    Too small (128 tokens)  precise retrieval, but the answer is split across
                            chunks and the model sees fragments
    Too large (2000 tokens) every chunk contains something about everything,
                            embeddings blur toward the document average, and
                            retrieval degrades to "return anything vaguely on topic"

Worse, a bad split is silent. Cut a table in half or separate a heading from
its content and retrieval gets quietly worse with no error anywhere. Chunking
bugs do not crash; they just make your product mediocre.

DEFAULTS THAT ARE USUALLY RIGHT

    ~512 tokens (≈2000 chars) with ~10-15% overlap, split on structure first.

Start there, then MEASURE with the evaluation module rather than guessing.
`examples/rag_ex_chunking_compare.py` prints the comparison.

WHY OVERLAP EXISTS

A fact that straddles a boundary is lost to both chunks. Overlap means the
sentence at a boundary appears in full in at least one chunk. The cost is
duplicate content (more storage, more embedding spend, and near-duplicate
results), so 10-20% is the usual compromise -- not 50%.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .types import Chunk, Document

# ~4 characters per token for English prose. Chunk sizes here are in CHARACTERS
# because that is what we can measure without a tokenizer dependency; divide by
# 4 for a rough token count.
CHARS_PER_TOKEN = 4


def chars_for_tokens(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


class Chunker(Protocol):
    """Structural interface. Any callable object splitting a Document works."""

    def split(self, document: Document) -> list[Chunk]: ...


def _finalize(
    pieces: Sequence[tuple[str, int]],
    document: Document,
    *,
    extra: Optional[dict] = None,
) -> list[Chunk]:
    """Turn (text, start_offset) pairs into Chunks, dropping empties."""
    chunks: list[Chunk] = []
    for text, start in pieces:
        stripped = text.strip()
        if not stripped:
            continue
        metadata = dict(document.metadata)
        if extra:
            metadata.update(extra)
        chunks.append(
            Chunk(
                text=stripped,
                doc_id=document.doc_id,
                source=document.source,
                ordinal=len(chunks),
                start=start,
                end=start + len(text),
                metadata=metadata,
            )
        )
    return chunks


@dataclass
class FixedSizeChunker:
    """Slice every N characters with a fixed overlap.

    The baseline. Fast, trivially predictable, and structure-blind: it will cut
    mid-word, mid-sentence, and mid-table without hesitation.

    Use it as a control when evaluating smarter strategies, or for genuinely
    unstructured text (OCR dumps, transcripts without punctuation). Do not ship
    it for documentation.
    """

    chunk_size: int = 2000
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap >= self.chunk_size:
            # Guarantees forward progress. With overlap >= chunk_size the
            # window never advances and you loop forever.
            raise ValueError("overlap must be < chunk_size")

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        step = self.chunk_size - self.overlap
        pieces = [(text[i : i + self.chunk_size], i) for i in range(0, len(text), step)]
        # Drop a trailing window only when its SPAN is already fully covered by
        # the previous window. Comparing text instead of positions looks
        # equivalent but silently drops real content on repetitive input -- a
        # final window of 'xxxx' is a substring of the previous chunk even
        # though it occupies different offsets.
        if len(pieces) > 1:
            last_text, last_start = pieces[-1]
            _, prev_start = pieces[-2]
            if last_start + len(last_text) <= prev_start + self.chunk_size:
                pieces.pop()
        return _finalize(pieces, document, extra={"chunker": "fixed"})


@dataclass
class RecursiveCharacterChunker:
    """Split on the largest natural boundary that fits. The sane default.

    This is the algorithm behind LangChain's RecursiveCharacterTextSplitter,
    implemented here so it is not magic.

    Try separators in order of semantic strength:

        "\\n\\n"  paragraph  -- strongest boundary
        "\\n"    line
        ". "    sentence
        " "     word
        ""      character   -- last resort

    Recurse: split on the strongest separator; any piece still too large gets
    split by the next separator down. The result respects meaning wherever it
    can and only degrades to brute force where it must.
    """

    chunk_size: int = 2000
    overlap: int = 200
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def __post_init__(self) -> None:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be < chunk_size")

    def _split_text(self, text: str, separators: Sequence[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # Keep the separator attached so rejoining reproduces the source.
        parts = text.split(separator)
        pieces = [p + separator for p in parts[:-1]] + [parts[-1]]

        out: list[str] = []
        buffer = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                # This single piece is oversized: flush and recurse on it.
                if buffer:
                    out.append(buffer)
                    buffer = ""
                out.extend(self._split_text(piece, rest))
            elif len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece  # merge small pieces up to the budget
            else:
                if buffer:
                    out.append(buffer)
                buffer = piece
        if buffer:
            out.append(buffer)
        return out

    def split(self, document: Document) -> list[Chunk]:
        raw = self._split_text(document.text, self.separators)

        # Apply overlap by prefixing each chunk with the tail of its predecessor.
        pieces: list[tuple[str, int]] = []
        cursor = 0
        for index, text in enumerate(raw):
            start = document.text.find(text, cursor)
            if start == -1:
                start = cursor
            if index > 0 and self.overlap:
                tail = raw[index - 1][-self.overlap :]
                pieces.append((tail + text, max(0, start - len(tail))))
            else:
                pieces.append((text, start))
            cursor = start + len(text)
        return _finalize(pieces, document, extra={"chunker": "recursive"})


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class MarkdownStructureChunker:
    """Split on Markdown headings and carry the heading path in metadata.

    The best strategy when your corpus has structure, which for documentation
    it always does. Two reasons it wins:

    1. A heading is an author-declared semantic boundary. Nothing you infer
       statistically beats a human saying "this is a new topic".
    2. The heading PATH disambiguates the chunk. "Employees get 25 days" is
       nearly useless in isolation; "Handbook > Benefits > Annual Leave:
       employees get 25 days" retrieves correctly and reads correctly.

    Oversized sections are delegated to the recursive chunker, with the heading
    path preserved on every resulting piece.
    """

    chunk_size: int = 2000
    overlap: int = 200
    prefix_heading: bool = True  # prepend the breadcrumb to chunk text

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return RecursiveCharacterChunker(self.chunk_size, self.overlap).split(document)

        sections: list[tuple[str, str, int]] = []  # (heading_path, body, start)
        stack: list[tuple[int, str]] = []  # (level, title)

        # Any content before the first heading still needs to be kept.
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()]
            if preamble.strip():
                sections.append(("", preamble, 0))

        for index, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)

            body_start = match.end()
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append((path, text[body_start:body_end], body_start))

        chunks: list[Chunk] = []
        for path, body, start in sections:
            if not body.strip():
                continue
            prefix = f"{path}\n" if (self.prefix_heading and path) else ""
            if len(body) + len(prefix) <= self.chunk_size:
                parts = [(prefix + body, start)]
            else:
                sub = RecursiveCharacterChunker(
                    self.chunk_size - len(prefix), self.overlap
                )._split_text(body, RecursiveCharacterChunker().separators)
                parts = []
                cursor = start
                for piece in sub:
                    parts.append((prefix + piece, cursor))
                    cursor += len(piece)

            for text_piece, piece_start in parts:
                if not text_piece.strip():
                    continue
                metadata = dict(document.metadata)
                metadata.update({"heading_path": path, "chunker": "markdown"})
                chunks.append(
                    Chunk(
                        text=text_piece.strip(),
                        doc_id=document.doc_id,
                        source=document.source,
                        ordinal=len(chunks),
                        start=piece_start,
                        end=piece_start + len(text_piece),
                        metadata=metadata,
                    )
                )
        return chunks


@dataclass
class SemanticChunker:
    """Split where the topic actually changes, measured by embedding distance.

    Embed each sentence, then cut wherever consecutive sentences are further
    apart than a percentile threshold of all the gaps in the document.

    Honest assessment: this is the most expensive strategy (one embedding per
    sentence at ingest) and it is NOT reliably better than a structure-aware
    split. It earns its cost on long unstructured prose -- transcripts,
    interviews, scanned reports -- where there are no headings to exploit.
    Reach for MarkdownStructureChunker first.

    `embed_fn` is injected so this stays offline and testable.
    """

    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]]
    chunk_size: int = 2000
    percentile: float = 0.80  # cut at gaps above this percentile

    def split(self, document: Document) -> list[Chunk]:
        sentences = _split_sentences(document.text)
        if len(sentences) < 3:
            return _finalize([(document.text, 0)], document, extra={"chunker": "semantic"})

        vectors = self.embed_fn([s for s, _ in sentences])
        distances = [1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = _percentile(distances, self.percentile)

        pieces: list[tuple[str, int]] = []
        buffer, buffer_start = "", sentences[0][1]
        for index, (sentence, offset) in enumerate(sentences):
            if not buffer:
                buffer_start = offset
            buffer += sentence
            at_breakpoint = index < len(distances) and distances[index] >= threshold
            if at_breakpoint or len(buffer) >= self.chunk_size:
                pieces.append((buffer, buffer_start))
                buffer = ""
        if buffer:
            pieces.append((buffer, buffer_start))
        return _finalize(pieces, document, extra={"chunker": "semantic"})


_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+\s*|[^.!?]+$")


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Sentences with their character offsets.

    A regex, not a real sentence tokenizer: it mishandles "Dr." and "e.g.".
    Adequate here, and the alternative is an nltk/spacy dependency for a
    teaching module. Production systems should use a proper segmenter.
    """
    return [(m.group(0), m.start()) for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def chunk_documents(documents: Sequence[Document], chunker: Chunker) -> list[Chunk]:
    out: list[Chunk] = []
    for document in documents:
        out.extend(chunker.split(document))
    return out


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    """Size distribution -- the first thing to look at when RAG underperforms.

    A large max/mean ratio means inconsistent chunks, which means inconsistent
    retrieval. Very small chunks are usually split artifacts.
    """
    if not chunks:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
    sizes = sorted(len(c) for c in chunks)
    return {
        "count": float(len(sizes)),
        "mean": sum(sizes) / len(sizes),
        "min": float(sizes[0]),
        "max": float(sizes[-1]),
        "p50": float(sizes[len(sizes) // 2]),
        "est_tokens": sum(sizes) / CHARS_PER_TOKEN,
    }
