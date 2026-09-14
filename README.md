# GenAI Backend Mastery

**From senior Java/Spring backend engineer → production GenAI engineer, in 12 months of part-time work.**

This is not a reading list. Every phase ships runnable code, a passing test suite, and something you can put in front of a hiring manager. The entire repo runs **offline with zero API keys and zero spend** — a deterministic `FakeProvider` stands in for OpenAI/Anthropic everywhere, so you can learn the mechanics before you pay for a single token.

```bash
git clone https://github.com/abhiparashar/genai-backend-mastery.git
cd genai-backend-mastery
make setup          # venv + dev deps
make test           # full offline suite. no keys, no cost.
```

---

## Why this repo exists

Job posts want "Backend Engineer + GenAI". The market is full of people who can call `openai.chat.completions.create()` in a notebook, and starving for people who can put an LLM behind an SLA — retries, budgets, streaming, evals, tenant isolation, cost attribution.

**You already own the hard half.** Distributed systems, resilience patterns, observability, API design, data modelling — that's the part most AI-curious candidates can't do. What you're missing is Python fluency and the GenAI-specific layer. That's exactly what this repo drills, and it constantly maps new concepts back to what you already know:

| You know (Java/Spring) | You'll learn (Python/GenAI) |
|---|---|
| Spring Boot | FastAPI |
| Bean Validation + Lombok | Pydantic v2 |
| `@Autowired` | `Depends()` |
| `@ConfigurationProperties` | `BaseSettings` |
| Resilience4j | `tenacity` + the circuit breaker in `llmkit` |
| Caffeine | `lru_cache` / the TTL cache in `llmkit` |
| `CompletableFuture` / `ExecutorService` | `asyncio.Task` / `gather` + `Semaphore` |
| Maven + `pom.xml` | `pip` + `pyproject.toml` |
| JUnit + Mockito | pytest + fixtures + `monkeypatch` |
| Micrometer + Actuator | the metrics + `/healthz` in `06-production-service` |
| Postgres + JPA | Postgres + **pgvector** (same DB, new index type) |

Full translation guide: [`best-practices/python-for-java-devs.md`](best-practices/python-for-java-devs.md).

---

## Repo map

| Directory | What's in it |
|---|---|
| [`01-python-foundations/`](01-python-foundations/) | ~60 exercises with stubs, solutions, and tests. Syntax → collections → OOP → exceptions. Written for someone who thinks in Java. |
| [`02-python-advanced/`](02-python-advanced/) | Decorators, generators, context managers, typing, and **asyncio** — the part that actually matters for LLM work. |
| [`03-llm-integration/`](03-llm-integration/) | `llmkit`: a provider-agnostic LLM client with retry+jitter, circuit breaker, fallback, caching, cost tracking, budget guards, and typed structured output. |
| [`04-rag/`](04-rag/) | `ragkit`: RAG built from scratch — 4 chunking strategies, BM25 + vector + **hybrid RRF**, reranking, citations, and RAGAS-style evaluation on a golden set. |
| [`05-agents/`](05-agents/) | `agentkit`: ReAct and native tool-calling loops by hand, JSON-Schema-from-type-hints, sandboxed tools, memory, planning, multi-agent, and guardrails. |
| [`06-production-service/`](06-production-service/) | The reference FastAPI service: SSE streaming, API-key auth, token-bucket rate limiting, PII redaction, metrics, Docker, Kubernetes manifests. |
| [`07-java-integration/`](07-java-integration/) | Two Spring Boot 3 projects: a WebFlux gateway calling the Python service (Resilience4j, SSE passthrough), and a Spring AI native RAG app. **Your unfair advantage.** |
| [`08-interview-prep/`](08-interview-prep/) | 100 concept answers, 10 worked system designs with capacity math, 10 tested coding challenges, behavioral STAR stories. |
| [`projects/`](projects/) | Three portfolio-grade builds: Document Q&A, multi-agent support, text-to-SQL. Plus a backlog of 30 more. |
| [`best-practices/`](best-practices/) | The opinionated standards: architecture, cost, security, testing non-determinism, observability, launch checklist. |

---

## The roadmap

**Assumption: 10–12 hours/week.** Double up and it's ~6 months; halve it and it's ~2 years. The phase order is load-bearing — each one consumes the last.

### Phase 0 — Setup (Week 0, 3 hours)

Install Python 3.11+ (`pyenv install 3.11.9`), clone this repo, `make setup`, `make test`. Read [`best-practices/python-for-java-devs.md`](best-practices/python-for-java-devs.md) end to end. Don't skip it — it prevents a month of writing Java in Python syntax.

> **Note:** this repo's floor is Python 3.9 (macOS system Python) so everything runs on a stock Mac, but **install 3.11+ anyway**. It's meaningfully faster and every employer is on it.

### Phase 1 — Python that isn't a crash course (Weeks 1–6)

`01-python-foundations/`

Work the exercises. Run `pytest 01-python-foundations/tests` — your unfinished exercises show as *skipped*, finished ones as *passed*. That's your progress bar.

Non-negotiable trapdoors (one exercise each): mutable default arguments, `is` vs `==`, shallow vs deep copy, late-binding closures in loops, EAFP vs LBYL, the `__eq__`/`__hash__` contract (your `equals`/`hashCode` instincts transfer), `@dataclass` vs `record`.

**Exit criteria:** you write comprehensions without thinking, you reach for `dict.get`/`defaultdict`/`Counter` reflexively, and you stop writing getters and setters.

### Phase 2 — The Python that GenAI actually needs (Weeks 7–12)

`02-python-advanced/`

Decorators (Spring AOP), generators (how token streaming works), context managers (try-with-resources), typing + `mypy`, and **asyncio**.

Spend the most time on async. An LLM call is 2–20 seconds of pure I/O wait. Sequential code wastes all of it; `asyncio.gather` with a `Semaphore` is the difference between a 40-second and a 3-second endpoint. The GIL means your Java threading instincts are actively wrong here — the module has a decision table.

**Exit criteria:** you can fan out 20 concurrent API calls with bounded concurrency, per-call timeouts, and partial-failure tolerance, and explain why threads wouldn't have helped.

### Phase 3 — LLM fundamentals + a client worth shipping (Weeks 13–20)

`03-llm-integration/` + [`best-practices/llm-app-architecture.md`](best-practices/llm-app-architecture.md)

Just enough theory (tokens, embeddings, context windows, temperature/top_p, why models hallucinate) then straight into engineering: the **failure-mode table** is the module. Which errors are retryable (429, 500, timeout) and which are fatal (401, context-length, content-filter), and what the client does about each.

You'll build exponential backoff with full jitter by hand, honor `Retry-After`, add a circuit breaker and provider fallback, cache safely (only at `temperature=0` — and you'll learn why), track cost per request, and enforce a daily budget ceiling that raises *before* the call.

Then `structured.py`: JSON mode → extract-from-prose repair → Pydantic validation → bounded re-ask with the validation errors fed back. This single technique carries more production weight than anything else in the module.

**Exit criteria:** `python 03-llm-integration/examples/llmkit_ex_resilience.py` — you can explain every retry, fallback, and breaker transition it prints.

### Phase 4 — RAG, built from scratch (Weeks 21–30)

`04-rag/`

The longest phase because RAG is most of the job market. **No LangChain yet** — you implement the recursive character splitter, BM25, cosine search, RRF fusion, and MMR yourself, so that when a framework misbehaves you know what it was supposed to do.

Then the parts tutorials skip: hybrid search beating pure vector on keyword queries (with numbers), reranking, metadata pre-filtering vs post-filtering, context-window budgeting, inline citations, the "I don't know" path below a similarity threshold, and **evaluation** — context precision/recall, faithfulness, hit rate, MRR, NDCG on a committed golden set.

> Untested RAG is the #1 reason GenAI features die in production. "Vibes-based retrieval" is a real failure mode with a real fix: a golden dataset and a scorecard in CI.

**Exit criteria:** you can show a scorecard, change the chunk size, and explain the metric movement.

### Phase 5 — Agents, and when not to use them (Weeks 31–38)

`05-agents/`

The ReAct loop by hand, then native tool calling. The genuinely hard parts: generating JSON Schema from Python type hints, sandboxing tools (AST-based calculator — never `eval`, path-jailed file reads, domain-allowlisted HTTP, read-only SQL), loop detection, mid-run budget enforcement, and **prompt injection via tool output** (a retrieved document that tells your agent to exfiltrate data).

The README's most valuable section is *when NOT to use an agent*. A 10-step agent run costs ~10× a single call and fails in 10× as many ways. Most "agent" problems are a chain with better prompts.

**Exit criteria:** an agent that terminates under every adversarial condition you throw at it, with a complete audit trail.

### Phase 6 — Production (Weeks 39–46)

`06-production-service/` + [`best-practices/production-checklist.md`](best-practices/production-checklist.md)

Everything you already do for Java services, applied to an LLM service: SSE streaming with correct framing and disconnect handling, API-key auth with constant-time compare, per-key token-bucket rate limiting with `Retry-After`, structured JSON logs with correlation IDs and **PII redaction**, Prometheus metrics (including tokens and dollars as first-class metrics), quota enforcement, multi-stage Dockerfile running non-root, and Kubernetes manifests with probes and an HPA.

**Exit criteria:** every box in the production checklist ticked, with the evidence.

### Phase 7 — The Java bridge (Weeks 47–50)

`07-java-integration/`

Where you stop competing with ML people and start competing on your own turf. A Spring Boot WebFlux gateway calling the Python AI service with Resilience4j (circuit breaker, retry, bulkhead, time limiter) and `Flux<ServerSentEvent<String>>` streaming passthrough. Then the same capability in **Spring AI**, no Python hop, with pgvector RAG.

Plus the decision table employers actually want you to have an opinion on: *what belongs in Java (auth, billing, business logic, transactions, orchestration) vs Python (LLM calls, RAG, embeddings, agents)* — and how to introduce GenAI into an existing monolith without a rewrite (strangler fig, feature flags, shadow traffic).

**Exit criteria:** you can whiteboard the hybrid architecture and defend every boundary.

### Phase 8 — Portfolio + interviews (Weeks 51–52+, ongoing)

`projects/` + `08-interview-prep/`

Ship the three projects, then keep going through the 30-item backlog. Work `08-interview-prep/` in parallel from Week 40 — 100 concept answers, 10 system designs with capacity math, 10 tested coding challenges, STAR stories built from the projects you just shipped.

---

## Accelerated tracks

| Your situation | Track |
|---|---|
| Interviewing in **6 weeks** | Phase 3 → 4 → `projects/01-document-qa` → `08-interview-prep`. Skim 1–2, but do the async file. |
| Interviewing in **3 months** | Phases 1–6 at double pace, all three projects, drop Phase 7. |
| **No deadline**, want depth | Full 12 months in order. Add the project backlog. |
| Already know Python | Start at Phase 3. Do `02-python-advanced/05_async.py` first. |

---

## How to prove you learned it

Reading produces nothing. Each phase has an artifact:

1. **Green tests** — `make test` stays green as you work.
2. **A running thing** — every module has examples you execute and read the output of.
3. **A public commit** — push your solutions. A GitHub history showing 12 months of consistent GenAI work *is* the portfolio.
4. **A written decision** — each project README documents alternatives you rejected and why. Interviewers probe exactly this.

**Anti-pattern to avoid:** collecting courses. You do not need 8 DeepLearning.AI certificates. You need three deployed projects and the ability to explain your retrieval evaluation.

---

## Working commands

```bash
make setup      # venv + dev dependencies
make test       # offline suite — no API keys, no spend
make test-live  # tests against real providers (costs money, needs .env)
make lint       # ruff check + format check
make types      # mypy
make check      # everything CI runs
make serve      # run the production FastAPI service locally
make optional   # heavy deps: real SDKs, vector DBs, frameworks
```

Going live with real providers:

```bash
cp .env.example .env     # add your keys
# set LLM_PROVIDER=openai (default is `fake`)
# LLM_DAILY_BUDGET_USD is a hard stop — keep it low while learning
```

> **Cost control while learning.** Default to `gpt-4o-mini` / `claude-haiku` class models; they're ~20× cheaper and adequate for everything here. The budget guard in `llmkit/cost.py` exists because a runaway agent loop at 3am is a real way to lose real money. The whole curriculum is completable for **under $20** of API spend.

---

## Ground rules baked into this repo

1. **Secrets never hit git.** `.env` is ignored; `.env.example` documents the shape.
2. **Tests never hit the network.** Unit tests mock the provider. Live tests are opt-in and marked.
3. **Every LLM call has a timeout, a retry policy, and a cost record.** No exceptions.
4. **Caching is only correct at `temperature=0`.** Cache a non-deterministic call and you've built a bug.
5. **Prompts are versioned files, not string literals.** You cannot A/B test a hardcoded string.
6. **Retrieved content is untrusted input.** Indirect prompt injection is the real attack.
7. **Evaluate before you optimize.** No golden set means no evidence, just vibes.

---

## Resources worth your time

Curated hard, because the field generates more noise than signal.

**Read the primary docs, completely:** [OpenAI API](https://platform.openai.com/docs), [Anthropic](https://docs.anthropic.com), [FastAPI](https://fastapi.tiangolo.com) (genuinely excellent), [Pydantic](https://docs.pydantic.dev), [Spring AI](https://docs.spring.io/spring-ai/reference/).

**Books:** *Designing Machine Learning Systems* — Chip Huyen (the systems thinking transfers directly); *AI Engineering* — Chip Huyen (closest thing to a textbook for this role).

**Courses (pick two, don't collect them):** DeepLearning.AI's *Building Systems with the ChatGPT API* and *Functions, Tools and Agents with LangChain*.

**Stay current without drowning:** Anthropic's and OpenAI's engineering blogs, Simon Willison's blog, and the LangChain/LlamaIndex changelogs. One hour a week, capped.

---

## Roadmap origin

Generated from a detailed curriculum brief (`roadmap-source.md` — 3,170 lines covering 118 weeks) and compressed into something executable. The source is kept for reference; this README is the version you actually follow.

## License

MIT — take it, fork it, make it yours.
