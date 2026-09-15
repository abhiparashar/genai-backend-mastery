"""Evaluating RAG: the part that separates engineering from vibes.

WHY THIS MODULE MATTERS MORE THAN ANY OTHER IN THE MODULE

Every RAG demo works. The question is whether yours works on the 200 questions
your users will actually ask, and whether the chunk-size change you just made
helped or hurt. Without measurement you cannot answer either, so teams tune RAG
by trying things and eyeballing three examples. That is not engineering.

The fix is unglamorous: a golden dataset of (question, expected answer,
relevant chunk ids) and a scorecard you can run in CI.

EVALUATE THE TWO HALVES SEPARATELY

This is the single most useful diagnostic idea here. A bad answer has two very
different possible causes, with opposite fixes:

    retrieval failed   -> the right chunk was never fetched
                          fix: chunking, embeddings, hybrid search, reranking
    generation failed  -> the right chunk WAS fetched and the model still
                          got it wrong
                          fix: the prompt, the model, context ordering

Measure `hit_rate` first. If retrieval is at 0.6, no prompt engineering will
save you, and every hour spent on the prompt is wasted.

THE METRICS

    Retrieval (need labelled relevant chunks)
      hit_rate@k       did ANY relevant chunk make the top k?      blunt, vital
      precision@k      what fraction of returned chunks are relevant?
      recall@k         what fraction of relevant chunks did we get?
      MRR              1/rank of the first relevant hit -- position matters
      NDCG@k           rank-weighted, handles graded relevance

    Generation (need an answer, and often a judge)
      faithfulness     is every claim supported by the retrieved context?
                       this is the hallucination metric
      answer_relevance does the answer address the question asked?
      correctness      does it match the expected answer?

FAITHFULNESS IS THE ONE THAT CATCHES HALLUCINATION. An answer can be relevant,
fluent, and completely invented. Faithfulness decomposes the answer into claims
and asks whether the context supports each one.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .embeddings import tokenize
from .types import Answer, Message, ScoredChunk, user

# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------


@dataclass
class GoldenExample:
    """One labelled test case.

    `relevant_chunk_texts` holds substrings that MUST appear in a retrieved
    chunk for it to count as relevant. Substrings rather than chunk ids on
    purpose: chunk ids change every time you change chunk size, which would
    invalidate your whole dataset exactly when you most need it -- while
    tuning chunking.
    """

    question: str
    expected_answer: str = ""
    relevant_chunk_texts: list[str] = field(default_factory=list)
    should_refuse: bool = False  # unanswerable questions belong in the set too
    metadata: dict[str, Any] = field(default_factory=dict)


def load_golden(path: str) -> list[GoldenExample]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [GoldenExample(**row) for row in data]


def save_golden(examples: Sequence[GoldenExample], path: str) -> None:
    payload = [
        {
            "question": e.question,
            "expected_answer": e.expected_answer,
            "relevant_chunk_texts": e.relevant_chunk_texts,
            "should_refuse": e.should_refuse,
            "metadata": e.metadata,
        }
        for e in examples
    ]
    pathlib.Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_relevant(scored: ScoredChunk, markers: Sequence[str]) -> bool:
    text = scored.text.lower()
    return any(marker.lower() in text for marker in markers)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def hit_rate(retrieved: Sequence[ScoredChunk], markers: Sequence[str]) -> float:
    """Did any relevant chunk appear at all? 1.0 or 0.0.

    The bluntest metric and the one to fix first. If hit_rate is low, nothing
    downstream can be good, because the information simply is not in the
    prompt.
    """
    if not markers:
        return 1.0
    return 1.0 if any(_is_relevant(c, markers) for c in retrieved) else 0.0


def precision_at_k(retrieved: Sequence[ScoredChunk], markers: Sequence[str]) -> float:
    """Fraction of returned chunks that are relevant.

    Low precision means you are paying tokens for noise, and diluting the
    model's attention with it.
    """
    if not retrieved:
        return 0.0
    return sum(1 for c in retrieved if _is_relevant(c, markers)) / len(retrieved)


def recall_at_k(retrieved: Sequence[ScoredChunk], markers: Sequence[str]) -> float:
    """Fraction of the known-relevant material that was retrieved."""
    if not markers:
        return 1.0
    found = sum(1 for m in markers if any(m.lower() in c.text.lower() for c in retrieved))
    return found / len(markers)


def mean_reciprocal_rank(retrieved: Sequence[ScoredChunk], markers: Sequence[str]) -> float:
    """1 / rank of the first relevant chunk.

    Position matters even when hit_rate is 1.0: models attend most reliably to
    the start of the context, so a relevant chunk at rank 1 is worth more than
    the same chunk at rank 8.

    >>> from .types import Chunk
    >>> hits = [ScoredChunk(Chunk("nope"), 1.0), ScoredChunk(Chunk("yes here"), 0.9)]
    >>> mean_reciprocal_rank(hits, ["yes"])
    0.5
    """
    for rank, scored in enumerate(retrieved, start=1):
        if _is_relevant(scored, markers):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[ScoredChunk], markers: Sequence[str]) -> float:
    """Normalized discounted cumulative gain -- rank-weighted relevance.

    Each relevant hit contributes 1/log2(rank+1), so gains decay smoothly with
    position rather than all-or-nothing. Normalized against the ideal ordering
    so 1.0 means "relevant chunks were ranked as well as possible".
    """
    if not markers or not retrieved:
        return 0.0
    gains = [1.0 if _is_relevant(c, markers) else 0.0 for c in retrieved]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(markers), len(retrieved))))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(answer: Answer, *, threshold: float = 0.5) -> float:
    """Fraction of the answer's claims supported by the retrieved context.

    THE HALLUCINATION METRIC. An answer can be fluent, relevant, and entirely
    invented; only this catches that.

    Implemented lexically: split the answer into sentences and check what
    fraction of each sentence's content words appear in the context. That is a
    weak proxy -- it cannot detect a claim that reuses context vocabulary to
    assert something the context does not say -- but it needs no LLM, runs in
    CI for free, and reliably catches wholesale fabrication.

    `llm_faithfulness` below is the accurate version. Use this one as a cheap
    always-on regression gate and that one for periodic deeper evaluation.
    """
    if not answer.chunks:
        return 0.0 if answer.text.strip() else 1.0

    context_tokens = set()
    for scored in answer.chunks:
        context_tokens.update(tokenize(scored.text))

    claims = _sentences(answer.text)
    if not claims:
        return 1.0

    supported = 0
    for claim in claims:
        claim_tokens = [t for t in tokenize(claim) if len(t) > 3]
        if not claim_tokens:
            supported += 1  # "Yes." carries no verifiable content
            continue
        overlap = sum(1 for t in claim_tokens if t in context_tokens) / len(claim_tokens)
        if overlap >= threshold:
            supported += 1
    return supported / len(claims)


def answer_relevance(answer: Answer, question: str) -> float:
    """Lexical overlap between the answer and the question's content words.

    Catches evasive or off-topic answers. Crude -- a good answer may share few
    words with the question -- so read it alongside faithfulness rather than
    alone.
    """
    question_tokens = {t for t in tokenize(question) if len(t) > 3}
    if not question_tokens:
        return 1.0
    answer_tokens = set(tokenize(answer.text))
    return len(question_tokens & answer_tokens) / len(question_tokens)


def answer_correctness(answer: Answer, expected: str) -> float:
    """Token F1 against the expected answer.

    F1 rather than exact match, because "25 days" and "Employees get 25 days
    of annual leave" are both correct and exact match would score the second
    one zero.
    """
    if not expected:
        return 1.0
    predicted = set(tokenize(answer.text))
    gold = set(tokenize(expected))
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def llm_faithfulness(complete: Callable[[Sequence[Message]], Any], answer: Answer) -> float:
    """LLM-as-judge faithfulness. Accurate, and not free.

    Calibrate it before trusting it: score 30 examples by hand, compare, and
    check the judge agrees with you. An uncalibrated judge is just a second
    opinion from the same kind of system that produced the answer -- and
    models are known to rate their own style of output favourably.
    """
    context = "\n\n".join(c.text for c in answer.chunks)
    prompt = (
        "Does the CONTEXT fully support every factual claim in the ANSWER?\n"
        "Reply with a single number 0-10, nothing else.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer.text}\n\nScore:"
    )
    reply = complete([user(prompt)])
    match = re.search(r"(\d+(?:\.\d+)?)", getattr(reply, "text", str(reply)))
    return min(1.0, float(match.group(1)) / 10.0) if match else 0.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    question: str
    hit_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    correctness: float = 0.0
    refused: bool = False
    refusal_correct: bool = True
    elapsed_ms: float = 0.0


@dataclass
class Scorecard:
    results: list[EvalResult] = field(default_factory=list)

    def _mean(self, attribute: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attribute) for r in self.results) / len(self.results)

    def summary(self) -> dict[str, float]:
        return {
            "n": float(len(self.results)),
            "hit_rate": self._mean("hit_rate"),
            "precision": self._mean("precision"),
            "recall": self._mean("recall"),
            "mrr": self._mean("mrr"),
            "ndcg": self._mean("ndcg"),
            "faithfulness": self._mean("faithfulness"),
            "answer_relevance": self._mean("relevance"),
            "correctness": self._mean("correctness"),
            "refusal_accuracy": (
                sum(1 for r in self.results if r.refusal_correct) / len(self.results)
                if self.results
                else 0.0
            ),
            "p50_latency_ms": (
                sorted(r.elapsed_ms for r in self.results)[len(self.results) // 2]
                if self.results
                else 0.0
            ),
        }

    def report(self) -> str:
        summary = self.summary()
        lines = [
            f"RAG scorecard over {int(summary['n'])} questions",
            "-" * 46,
            "RETRIEVAL  (fix these first -- no prompt saves bad retrieval)",
            f"  hit_rate          {summary['hit_rate']:.2f}",
            f"  precision@k       {summary['precision']:.2f}",
            f"  recall@k          {summary['recall']:.2f}",
            f"  MRR               {summary['mrr']:.2f}",
            f"  NDCG@k            {summary['ndcg']:.2f}",
            "GENERATION",
            f"  faithfulness      {summary['faithfulness']:.2f}   <- hallucination",
            f"  answer_relevance  {summary['answer_relevance']:.2f}",
            f"  correctness (F1)  {summary['correctness']:.2f}",
            f"  refusal_accuracy  {summary['refusal_accuracy']:.2f}",
            f"  p50 latency       {summary['p50_latency_ms']:.0f} ms",
        ]
        return "\n".join(lines)

    def failures(self, metric: str = "hit_rate", below: float = 1.0) -> list[EvalResult]:
        """The questions worth actually reading. Averages hide the failures."""
        return [r for r in self.results if getattr(r, metric) < below]


def evaluate(
    pipeline: Any,
    examples: Sequence[GoldenExample],
    *,
    judge: Optional[Callable[[Sequence[Message]], Any]] = None,
) -> Scorecard:
    """Run a pipeline against a golden set and score every dimension."""
    card = Scorecard()
    for example in examples:
        answer = pipeline.answer(example.question)
        retrieved = answer.chunks
        markers = example.relevant_chunk_texts

        result = EvalResult(
            question=example.question,
            hit_rate=hit_rate(retrieved, markers),
            precision=precision_at_k(retrieved, markers),
            recall=recall_at_k(retrieved, markers),
            mrr=mean_reciprocal_rank(retrieved, markers),
            ndcg=ndcg_at_k(retrieved, markers),
            faithfulness=(llm_faithfulness(judge, answer) if judge else faithfulness(answer)),
            relevance=answer_relevance(answer, example.question),
            correctness=answer_correctness(answer, example.expected_answer),
            refused=answer.refused,
            refusal_correct=(answer.refused == example.should_refuse),
            elapsed_ms=answer.elapsed_ms,
        )
        card.results.append(result)
    return card


def compare(cards: dict[str, Scorecard]) -> str:
    """Side-by-side comparison of configurations. This is how you tune RAG.

    Change ONE thing (chunk size, hybrid on/off, top_k), re-run, and read the
    numbers. Not "this feels better".
    """
    if not cards:
        return "(no results)"
    metrics = ["hit_rate", "precision", "mrr", "ndcg", "faithfulness", "correctness"]
    width = max(len(name) for name in cards) + 2

    header = "config".ljust(width) + "".join(m[:12].rjust(14) for m in metrics)
    lines = [header, "-" * len(header)]
    for name, card in cards.items():
        summary = card.summary()
        lines.append(name.ljust(width) + "".join(f"{summary[m]:.2f}".rjust(14) for m in metrics))
    return "\n".join(lines)
