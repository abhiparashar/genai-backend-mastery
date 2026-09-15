# Module 06 — Production Service

**Phase 6 of the roadmap · Weeks 39–46**

The reference FastAPI service. This is the module a hiring manager reads to decide whether you're senior — everything you already do for a Java service, applied to an LLM.

**It boots and serves with no API key.**

```bash
make serve                                    # or:
uvicorn app.main:app --app-dir 06-production-service --reload

curl localhost:8000/healthz
curl -X POST localhost:8000/v1/chat -H 'X-API-Key: dev-key-1' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'

pytest 06-production-service/tests -q         # 43 tests, offline
```

---

## Spring Boot ↔ FastAPI

| Spring Boot | Here |
|---|---|
| `@RestController` | `APIRouter` |
| `@RequestBody` + `@Valid` + DTO | Pydantic model (validates *and* generates OpenAPI) |
| `@ConfigurationProperties` | `BaseSettings` — `config.py` |
| `@Autowired` | `Depends()` — `deps.py` |
| `Filter` / `HandlerInterceptor` | middleware — `middleware.py` |
| `@ExceptionHandler` | `app.exception_handler` — `main.py` |
| Actuator `/health` | `/healthz`, `/readyz` |
| Micrometer + Prometheus | `observability.py` |
| Resilience4j `RateLimiter` | `TokenBucket` |
| `@PostConstruct` / `@PreDestroy` | `lifespan` |
| Logback JSON encoder | `JSONFormatter` |
| Springdoc OpenAPI | built in — `/docs` |

---

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/chat` | Sync completion |
| `POST` | `/v1/chat/stream` | **SSE** — named events |
| `POST` | `/v1/chat/async` | → `202` + `job_id` |
| `GET` | `/v1/jobs/{id}` | Poll job state |
| `GET` | `/healthz` | Liveness — checks **nothing** |
| `GET` | `/readyz` | Readiness — checks dependencies, `503` when degraded |
| `GET` | `/metrics` | Prometheus text |
| `GET` | `/docs` | Swagger UI |

Auth on every `/v1` route: `X-API-Key`. Optional `X-Tenant-ID` drives cost attribution; `X-Correlation-ID` is accepted and echoed.

### One error envelope, everywhere

```json
{"error": {"code": "rate_limited", "message": "…", "correlation_id": "a1b2c3"}}
```

| Status | `code` | Meaning |
|---|---|---|
| 401 | `unauthorized` | missing/bad key — deliberately doesn't say which |
| 413 | `payload_too_large` | rejected on `Content-Length`, before reading |
| 422 | `invalid_request` | validation, with the offending field |
| 429 | `rate_limited` | + `Retry-After`, `X-RateLimit-*` |
| **402** | `budget_exceeded` | **out of money — retrying will not help** |
| 502/504 | `upstream_error` | provider failed or timed out |
| 500 | `internal` | generic; the traceback goes to logs only |

---

## Decisions worth defending in an interview

### Liveness ≠ readiness

`/healthz` checks **nothing**. If it checked Redis, a 10-second Redis blip fails the liveness probe and Kubernetes **restarts every pod simultaneously** — turning degradation into an outage.

`/readyz` *does* check dependencies. Failing it removes the pod from the load balancer without killing it, so it rejoins automatically.

> Liveness answers *"restart me?"*. Readiness answers *"route to me?"*.

### Token bucket, not fixed window

| Algorithm | Why not |
|---|---|
| Fixed window | allows 2× the limit across a boundary (60 at 11:59:59 + 60 at 12:00:00) |
| Sliding log | exact, but O(N) memory per key |
| Leaky bucket | no burst tolerance at all |
| **Token bucket** | **O(1) memory, and tolerates bursts** |

Burst tolerance decides it: a page load fires six requests at once. A limiter that rejects that is technically correct and practically useless.

Single-process: each replica has its own bucket, so N replicas allow N× the limit. Production needs Redis. Stated rather than hidden.

### 402, not 429, for an exhausted budget

`429` means *slow down and retry*. A well-behaved client will — forever. `402 Payment Required` means *retrying will not help*.

And there are **two** limits, because they fail differently:

- **Global budget** protects you.
- **Per-tenant quota** protects you from *one* customer. Without it, one abusive client drains the global budget and takes the service down for everyone — a billing problem becomes an availability incident.

Both enforced **before** the call. A check afterwards is an invoice.

### SSE, not WebSocket

Traffic is one-directional: the server streams, the client listens. SSE is plain HTTP, so it keeps proxies, load balancers, auth headers and HTTP/2. WebSocket adds bidirectionality you don't need plus an upgrade handshake infrastructure often mishandles.

Three things people get wrong, all handled here:

1. **Framing** — `data: <payload>\n\n`. The blank line is the delimiter; omit it and the client buffers forever.
2. **Disconnects** — if the client vanishes, stop calling the provider. Otherwise you pay for tokens nobody receives.
3. **Errors after the first byte** — `200 OK` is already sent, so the status can't change. Errors travel in-band as an `error` event; the alternative is a silent truncation.

```
event: start   data: {"id":"...","model":"gpt-4o-mini"}
event: token   data: {"delta":"Refunds "}
event: usage   data: {"usage":{...},"cost_usd":0.000012}
event: done    data: [DONE]
```

Plus `X-Accel-Buffering: no` — without it nginx buffers the whole response and streaming works locally while silently failing in production.

### Middleware order is load-bearing

Correlation ID outermost (so every later log line has it) → logging → body-size (reject before reading) → **rate limit before auth** (a token-bucket check is cheaper than a credential comparison, so an unauthenticated flood costs less).

---

## What gets measured

Everything a Java service measures, plus the two this domain adds:

**`llm_input_tokens_total`, `llm_output_tokens_total`, `llm_cost_usd_total`, `llm_cost_usd_by_tenant`**

Cost is a first-class metric, not a monthly finance report. If you can't answer *"what did the last hour cost, and which tenant caused it?"* from a dashboard, you'll find out from the bill.

**Percentiles, never means.** LLM latency is heavily right-skewed. A mean of 900ms hides a p99 of 30s — and the p99 is what users complain about.

**Correlation IDs** are accepted from upstream, not regenerated, so a trace started in the Java gateway (module 07) continues here.

**Redaction runs at the logging boundary**, as a `logging.Filter` — not at call sites. Relying on every developer to remember is how a customer's email sits in CloudWatch for seven years and is found during an audit.

---

## Production readiness checklist

| | Item | Evidence |
|---|---|---|
| ✅ | Auth on every data route | `require_api_key`, constant-time compare |
| ✅ | Rate limiting | token bucket + `Retry-After` |
| ✅ | Input validation | Pydantic; model allowlist; size + repetition checks |
| ✅ | Body size limit | rejected on `Content-Length` |
| ✅ | Spend ceiling | global + per-tenant, enforced pre-flight |
| ✅ | Timeouts | `asyncio.wait_for` on every provider call |
| ✅ | Structured logs | JSON + correlation ID + PII redaction |
| ✅ | Metrics | Prometheus, incl. tokens and dollars |
| ✅ | Health probes | liveness and readiness, correctly distinct |
| ✅ | Error hygiene | one envelope; no stack traces to clients |
| ✅ | Graceful shutdown | `lifespan` + 60s grace period |
| ✅ | Container hardening | non-root, read-only rootfs, dropped caps |
| ✅ | Tests | 43, offline, no keys |
| ⚠️ | Distributed state | rate limits/budgets/jobs are **in-process** → Redis |
| ⚠️ | Response caching | see `llmkit/cache.py` |
| ⚠️ | Tracing | OpenTelemetry not wired |

The three ⚠️ rows are honest gaps, not oversights. A checklist with everything ticked is usually a checklist nobody read.

---

## Load test

```bash
pip install locust
```

```python
# locustfile.py
from locust import HttpUser, task, between


class ChatUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat(self):
        self.client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"X-API-Key": "dev-key-1"},
        )
```

```bash
locust -f locustfile.py --host http://localhost:8000
```

Watch `/metrics` for `p99` and `http_rate_limited_total` as you ramp.

---

## Exit criteria

1. You can explain why `/healthz` must not check the database.
2. You can justify token bucket over the other three algorithms.
3. You can name the three SSE mistakes and show where each is handled.
4. You can explain 402 vs 429 for an exhausted quota.
5. You can point at every ✅ above and name the file that earns it.

**Next:** [Module 07 — Java Integration](../07-java-integration/) — the Spring Boot gateway that calls this service.
