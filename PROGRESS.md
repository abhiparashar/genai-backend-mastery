# Build Progress Tracker

**Purpose:** this file is the handoff document. Any future session (human or AI) reads
this first, and can continue building the repo without re-deriving decisions.

Last updated: 2026-09-16 · Repo: https://github.com/abhiparashar/genai-backend-mastery

---

## How we work

- **One file at a time.** Write a file → smoke-test it → lint it → `git commit` → `git push`.
  Never batch multiple modules into one commit. Small commits survive interruptions.
- **Never leave a stub.** If a file is committed, it is finished. Unstarted work lives in
  the checklist below, not as `TODO` comments in code.
- **Verify before committing.** Every module gets an actual execution, not an eyeball:
  `.venv/bin/python -c "..."` smoke test, then `.venv/bin/ruff check <path>`.

### Commands

```bash
cd ~/Desktop/genai-backend-mastery
.venv/bin/python -m pytest -m "not live" -q     # full offline suite
.venv/bin/ruff check <path> && .venv/bin/ruff format <path>
git add -A && git commit -m "..." && git push origin main
```

If `.venv` is missing: `make setup`.

---

## Writing style (match this exactly)

The repo has one voice. New files must be indistinguishable from existing ones.

**Audience:** a senior Java/Spring backend engineer with zero Python and zero ML.
Never explain what a REST API or a circuit breaker is. Always explain what a decorator
or the GIL is.

**Module docstring.** Every file opens with one. It states WHAT the module does and WHY
it exists — the design rationale, not a feature list. Where a module carries the
module's central idea, say so outright (see `llmkit/errors.py`: *"THIS MODULE IS THE
LIBRARY. Everything else is plumbing."*).

**Comments earn their place.** They explain the non-obvious: a failure mode, a cost, a
trap, why the naive version is wrong. Never restate the code.

```python
# GOOD - explains a trap the reader would hit
# Still capped: a hostile or buggy Retry-After of 3600 must not hang us.
return min(float(retry_after), policy.max_delay)

# BAD - restates the code
# return the minimum of retry_after and max_delay
```

**Java analogies, used surgically.** One per concept, at first introduction, then move
on. `Pydantic == Bean Validation + Lombok`, `Depends() == @Autowired`,
`tenacity == Resilience4j`, `asyncio.Task == CompletableFuture`. Do not pile them up.

**Teach the trade-off, not just the API.** Any time there is a choice, name the
alternative and why it lost. The reader is being trained to defend decisions in an
interview.

**Numbers over adjectives.** "~4 chars per token", "an LLM call is 2-20s of I/O",
"a 10-step agent run costs ~10x a single call" — not "LLM calls are slow".

**Docstrings** use the imperative contract style, with doctest-style examples where the
function is pure and the example is short:

```python
def compute_delay(attempt: int, policy: RetryPolicy) -> float:
    """Delay before the next attempt. `attempt` is 0-based.

    >>> p = RetryPolicy(base_delay=1.0, jitter=False)
    >>> [compute_delay(i, p) for i in range(4)]
    [1.0, 2.0, 4.0, 8.0]
    """
```

**Commit messages:** `type(scope): summary`, then a body explaining *why*, with a
per-file line when the commit touches several. See `git log` for the pattern.

---

## Hard technical constraints

These are load-bearing. Violating them breaks CI or breaks the repo's promise.

1. **Python 3.9 is the floor** (stock macOS python3). 3.11 is the recommended target.
   Every `.py` file starts with `from __future__ import annotations`.
2. **`X | None` is BANNED in runtime-evaluated positions.** Pydantic v2 and FastAPI
   evaluate annotations at runtime even with the `__future__` import, and on 3.9 that
   raises. Use `Optional[X]` / `Union[...]` in model fields and route handlers.
   Ruff rules `UP007`/`UP045` are disabled in `pyproject.toml` for this reason.
   Builtin generics (`list[str]`, `dict[str, Any]`) ARE fine — they only ever appear in
   lazily-evaluated annotations. `UP006` stays enforced.
   Also banned: `match`, `tomllib`, `itertools.pairwise`, `typing.Self`.
3. **Core deps only**: stdlib, `pydantic`, `pydantic-settings`, `httpx`, `tenacity`,
   `python-dotenv`, `fastapi`, `uvicorn`, `numpy`. Dev: `pytest`, `pytest-asyncio`,
   `respx`, `ruff`, `mypy`.
   Everything else (`openai`, `anthropic`, `tiktoken`, `chromadb`,
   `sentence-transformers`, `pypdf`, `redis`, `sqlalchemy`, `langchain`) is **optional**:
   guarded import + `pytest.importorskip` in its tests. They are NOT installed.
4. **Zero network in tests.** The suite runs offline with no API keys. Real-provider
   tests are marked `@pytest.mark.live` and deselected by default. Mock HTTP with `respx`.
5. **Test file names are globally unique** (no `__init__.py` in test dirs, so duplicate
   basenames crash collection). Prefix with the module slug: `test_llmkit_retry.py`,
   `test_rag_chunking.py`.
6. **Every `tests/` dir needs this `conftest.py`** (dirs start with digits, so they are
   not importable packages):
   ```python
   from __future__ import annotations
   import pathlib, sys

   sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
   ```
7. **Line length 100**, ruff-formatted.

---

## The shared LLM contract

Modules 03/04/05/06 and every project are **standalone** — each defines its own copy of
this shape rather than importing across directories (directory names starting with
digits are not importable). Keep the field names IDENTICAL everywhere so the repo reads
as one system. Canonical definition: `03-llm-integration/llmkit/types.py`.

```python
Message(role, content)                      # role: system|user|assistant|tool
Usage(input_tokens, output_tokens)          # .total_tokens, supports +
ToolCall(id, name, arguments)
LLMResponse(text, usage, model, finish_reason, latency_ms, cached, tool_calls, cost_usd)
LLMProvider: .complete() / .acomplete() / .stream()
```

`FakeProvider` is the backbone of every test: deterministic, seeded, scriptable canned
replies, `fail_times=N` then succeed, artificial latency, call counter.

### Cross-service HTTP contract (module 06 Python service ⇄ module 07 Java gateway)

Decided and final. Module 07's Spring gateway is built against this.

- Auth `X-API-Key`. Correlation `X-Correlation-ID`, echoed on every response.
- `POST /v1/chat` → `{text, model, finish_reason, usage{input_tokens,output_tokens,total_tokens}, cost_usd, latency_ms, cached, correlation_id}`
- Request body is **`messages[]` + `conversation_id`** (not `prompt`+`system` — chosen
  because it maps 1:1 onto both provider payloads and is multi-turn native).
- `POST /v1/chat/stream` → SSE with **named events**:
  `start{id,model}` · `token{delta}` · `heartbeat{ts}` · `usage{usage,cost_usd}` ·
  `error{code,message}` · `done` with data `[DONE]`.
- `POST /v1/chat/async` → 202 `{job_id,status}`; `GET /v1/jobs/{id}` → status/result.
- `GET /healthz` (liveness) · `/readyz` (503 when a dep is down) · `/metrics` (Prometheus).
- Error envelope on every non-2xx:
  `{"error":{"code":"rate_limited|invalid_request|unauthorized|budget_exceeded|upstream_error|internal","message":"...","correlation_id":"..."}}`
- Statuses: 401 · 413 · 422 · 429 (+`Retry-After`, `X-RateLimit-*`) · 402 budget · 502/503 upstream · 500 internal.
- Resilience: retry ONLY 429/502/503/504 + connect/read timeouts; **never** 400/401/402/413/422.
  4xx must not open the circuit breaker. Gateway time limiter 35s vs service timeout 30s.
  No retry on a stream after the first byte is sent.

---

## Status

Legend: `[x]` done & pushed · `[~]` in progress · `[ ]` not started

### Done

- [x] Repo scaffold — `pyproject.toml` (ruff/pytest/mypy), `Makefile`, `.gitignore`,
      `.env.example`, `requirements{,-dev,-optional}.txt`
- [x] `.github/workflows/ci.yml` — offline matrix on 3.9/3.11/3.12, lint + format + tests + mypy
- [x] `README.md` — the 8-phase / 12-month roadmap, Java↔Python mapping, accelerated tracks
- [x] `roadmap-source.md` — the original 3,169-line curriculum brief (reference)
- [x] **Module 03 `03-llm-integration/` — COMPLETE.** 51 tests, 0.3s, fully offline.
  - `llmkit/types.py` — Message/Usage/ToolCall/LLMResponse/LLMProvider, `split_system()`
  - `llmkit/errors.py` — retryable-vs-fatal taxonomy, `classify_status()`
  - `llmkit/retry.py` — full-jitter backoff, Retry-After, wall-clock budget, sync+async
  - `llmkit/providers/fake.py` — FakeProvider (scripted replies, `fail_times`, latency, counters)
  - `llmkit/cost.py` — price table + `LAST_VERIFIED`, `CostTracker`, pre-flight `BudgetGuard`
  - `llmkit/cache.py` — stable key, LRU+TTL, `RedisCache`, temperature-0 guard
  - `llmkit/circuit.py` — CLOSED/OPEN/HALF_OPEN, injectable clock
  - `llmkit/providers/openai.py` — httpx, sync+async+SSE+embeddings
  - `llmkit/providers/anthropic.py` — Messages API, exposes the 5 wire diffs
  - `llmkit/structured.py` — 4-layer typed output with bounded re-ask
  - `llmkit/client.py` — façade: cache→budget→breaker→retry→provider→cost
  - `llmkit/__init__.py` — 40 public exports, lazy provider loading
  - `tests/test_llmkit_resilience.py` (23) + `tests/test_llmkit_client.py` (28)
  - `examples/llmkit_ex_resilience.py`, `examples/llmkit_ex_concurrent.py` (12.1x measured)
  - `README.md` — architecture, failure-mode table, wire-diff table, exit criteria

- [x] **Module 04 `04-rag/` — COMPLETE.** 64 tests, 0.4s, fully offline.
  - `ragkit/types.py` — Document/Chunk/ScoredChunk/Answer + standalone FakeProvider
  - `ragkit/chunking.py` — fixed / recursive / markdown-structure / semantic + stats
  - `ragkit/embeddings.py` — HashingEmbedder (offline), SemanticStubEmbedder
    (teaching device), sentence-transformers + OpenAI (guarded), CachingEmbedder
  - `ragkit/vectorstore.py` — exact cosine, IVFIndex (measured recall), Chroma
    (guarded), pre-filtering, tenant isolation, memory estimates
  - `ragkit/retrieval.py` — BM25, vector, hybrid RRF, MMR, LLM + cross-encoder
    rerankers, multi-query, HyDE, parent-child
  - `ragkit/pipeline.py` — ingest/answer, citations, context budget, refusal path
  - `ragkit/evaluation.py` — golden sets, hit_rate/precision/recall/MRR/NDCG,
    faithfulness, correctness, Scorecard + compare()
  - `data/` — 6-doc corpus + 16-question golden set (2 unanswerable)
  - `tests/test_rag_chunking.py` (24) + `tests/test_rag_retrieval.py` (40)
  - `examples/rag_ex_hybrid_vs_vector.py`, `examples/rag_ex_evaluate.py`
  - `README.md` — chunking tension, hybrid table, troubleshooting, exit criteria

### Next up (in order)

- [x] **Module 05 `05-agents/` — COMPLETE.** 80 tests, <1s, fully offline.
  - `agentkit/types.py` — ToolCall/ToolResult/Step/AgentResult, STOP_REASONS,
    scriptable FakeProvider
  - `agentkit/tools.py` — @tool JSON-Schema derivation from type hints +
    docstring; sandboxed calculator (AST), read_file (path jail), http_get
    (allowlist), sql_query (read-only); ToolRegistry that never raises
  - `agentkit/react.py` — the loop, forgiving parser, five termination
    guarantees (max_iterations/loop_detected/budget/deadline/no_progress)
  - `agentkit/memory.py` — Buffer/Window/TokenWindow/Summary/Vector +
    ConversationBudget, all pinning the system message
  - `agentkit/guardrails.py` — indirect-injection detection + neutralize,
    PII redaction (Luhn-checked), allowlist/denylist, fail-closed human
    approval, audit trail
  - `tests/test_agent_guardrails.py` (25) + `tests/test_agent_tools_and_loop.py` (55)
  - `examples/agent_ex_react.py` — multi-step, SQL, error recovery,
    termination, indirect injection
  - `README.md` — when NOT to use an agent, termination table, sandbox table,
    injection defence ordering, memory tradeoffs, exit criteria

- [x] **Module 06 `06-production-service/` — COMPLETE.** 43 tests, offline, boots with no key.
  - `app/config.py` — pydantic-settings, SecretStr, startup_check warnings
  - `app/llm.py` — offline FakeProvider + httpx OpenAI, degrades without a key
  - `app/schemas.py` — model allowlist, bounded temperature/max_tokens
  - `app/security.py` — constant-time key compare, size/repetition limits,
    advisory injection heuristics, output secret scan
  - `app/observability.py` — JSON logs, correlation ContextVar, redaction
    filter, percentile metrics, Prometheus exposition
  - `app/costs.py` — global + per-tenant windows, pre-flight enforcement
  - `app/middleware.py` — correlation id, request logging, body-size limit,
    token-bucket rate limiting with Retry-After
  - `app/deps.py` — Depends() wiring, Principal, test hooks
  - `app/routers/{chat,health}.py` — sync, SSE stream, async jobs, probes, metrics
  - `app/main.py` — app factory, lifespan, one error envelope
  - `Dockerfile` + `.dockerignore` + `docker-compose.yml` + `k8s/` (7 YAML docs)
  - `tests/test_service_api.py` (43)
  - `README.md` — Spring<->FastAPI map, decision rationale, readiness checklist

**Optional polish (skip unless wanted — modules 03/04/05/06 are usable now)**
- [ ] `03-llm-integration/examples/llmkit_ex_{basic,streaming,structured}.py`
- [ ] `04-rag/examples/rag_ex_{minimal,chunking_compare}.py`

**Then, in this order**
- [ ] `07-java-integration/` — `spring-ai-gateway/` (WebFlux + Resilience4j + SSE passthrough) and `spring-ai-native/` (Spring AI + pgvector RAG) + docker-compose + README (Java-vs-Python boundary decision table, strangler-fig migration)
- [ ] `01-python-foundations/` — 10 exercise files / ~60 exercises, stub+solution+test triplets, Java-trapdoor coverage
- [ ] `02-python-advanced/` — 6 topic files: decorators, generators, context managers, typing, **async** (highest value), performance
- [ ] `08-interview-prep/` — 100 concept answers, 10 worked system designs w/ capacity math, 10 tested coding challenges, behavioral STAR, java-challenges
      (verbatim question text lives in `roadmap-source.md`: concepts 2767-2887, designs 2889-2988, framework 2989-3013, architecture Qs 3014-3030, python challenges 3034-3044, java 3046-3056, behavioral 3064-3075, take-homes 3108-3128)
- [ ] `best-practices/` — 9 docs: index, python-for-java-devs, llm-app-architecture, cost-optimization, security, testing-genai, observability, production-checklist, dependency-management
- [ ] `projects/` — `01-document-qa`, `02-support-agent`, `03-text-to-sql`, each with README/src/tests/Dockerfile/compose/demo data + `projects/README.md` index with the 30-project backlog

### Housekeeping owed at the end

- [ ] Root `README.md` links every module README once they exist
- [ ] `make check` green (lint + format + mypy + full offline suite)
- [ ] Confirm CI passes on GitHub Actions
