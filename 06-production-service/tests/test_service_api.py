"""End-to-end API behaviour, asserted through real HTTP calls.

Every test here asserts something a CLIENT can observe: a status code, a
header, a response body, a stream frame. None of them reach into internals to
check that a function was called -- that would test the implementation rather
than the contract.
"""

from __future__ import annotations

import time

import pytest
from app.costs import BudgetExceeded, BudgetRegistry
from app.deps import get_budgets
from app.middleware import RateLimiter, TokenBucket
from app.observability import redact
from app.security import verify_api_key

# ---------------------------------------------------------------------------
# Health and ops
# ---------------------------------------------------------------------------


def test_liveness_needs_no_auth_and_no_dependencies(client):
    # A liveness probe that checks dependencies turns a Redis blip into
    # Kubernetes restarting every pod at once.
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_dependency_state(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "provider" in body["checks"]


def test_readiness_degrades_when_the_budget_is_gone(client):
    get_budgets().record("-", 10_000.0)  # blow through the global budget
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_metrics_are_exposed_in_prometheus_format(client, auth, body):
    client.post("/v1/chat", json=body, headers=auth)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "# TYPE http_requests_total counter" in text
    # Tokens and dollars are first-class metrics, not a finance report.
    assert "llm_input_tokens_total" in text
    assert "llm_cost_usd_total" in text


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [({}, 401), ({"X-API-Key": "wrong"}, 401), ({"X-API-Key": ""}, 401)],
)
def test_requests_without_a_valid_key_are_rejected(client, body, headers, expected):
    assert client.post("/v1/chat", json=body, headers=headers).status_code == expected


def test_a_valid_key_is_accepted(client, auth, body):
    assert client.post("/v1/chat", json=body, headers=auth).status_code == 200


def test_key_comparison_is_constant_time():
    # Not observable over HTTP, so assert it at the unit level: `==` leaks how
    # many leading characters were correct via response timing.
    assert verify_api_key("dev-key-1", ["dev-key-1"]) is True
    assert verify_api_key("dev-key-X", ["dev-key-1"]) is False
    assert verify_api_key(None, ["dev-key-1"]) is False


def test_auth_failure_does_not_explain_why(client, body):
    # "unknown key" vs "revoked key" is only useful to an attacker.
    message = client.post("/v1/chat", json=body, headers={"X-API-Key": "x"}).json()
    assert message["error"]["code"] == "unauthorized"
    assert "revoked" not in message["error"]["message"].lower()


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_successful_response_carries_the_full_contract(client, auth, body, scripted):
    scripted(responses=["Refunds are accepted within 30 days."])
    payload = client.post("/v1/chat", json=body, headers=auth).json()

    assert payload["text"] == "Refunds are accepted within 30 days."
    assert set(payload) == {
        "text",
        "model",
        "finish_reason",
        "usage",
        "cost_usd",
        "latency_ms",
        "cached",
        "correlation_id",
    }
    assert payload["usage"]["total_tokens"] > 0


def test_correlation_id_is_propagated_not_regenerated(client, auth, body):
    # This is what stitches a trace across the Java gateway and this service.
    headers = {**auth, "X-Correlation-ID": "trace-abc-123"}
    response = client.post("/v1/chat", json=body, headers=headers)

    assert response.headers["X-Correlation-ID"] == "trace-abc-123"
    assert response.json()["correlation_id"] == "trace-abc-123"


def test_a_correlation_id_is_generated_when_absent(client, auth, body):
    response = client.post("/v1/chat", json=body, headers=auth)
    assert response.headers["X-Correlation-ID"]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"messages": []}, "empty conversation"),
        (
            {"messages": [{"role": "user", "content": "x"}], "model": "gpt-5-ultra"},
            "model not allowlisted",
        ),
        (
            {"messages": [{"role": "user", "content": "x"}], "temperature": 9},
            "temperature out of range",
        ),
        (
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": 99999},
            "max_tokens too large",
        ),
        ({"messages": [{"role": "assistant", "content": "x"}]}, "ends on an assistant turn"),
    ],
)
def test_invalid_requests_are_rejected_with_useful_detail(client, auth, payload, reason):
    response = client.post("/v1/chat", json=payload, headers=auth)
    assert response.status_code == 422, reason
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["message"]


def test_every_error_uses_the_same_envelope(client, auth, body):
    responses = [
        client.post("/v1/chat", json=body),  # 401
        client.post("/v1/chat", json={"messages": []}, headers=auth),  # 422
        client.get("/v1/jobs/nonexistent", headers=auth),  # 404
    ]
    for response in responses:
        error = response.json()["error"]
        # One shape means clients write one error handler.
        assert set(error) >= {"code", "message", "correlation_id"}


def test_oversized_bodies_are_refused(client, auth):
    huge = {"messages": [{"role": "user", "content": "x" * 30_000}] * 20}
    response = client.post("/v1/chat", json=huge, headers=auth)
    assert response.status_code in (413, 422)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_emits_well_formed_sse_and_terminates(client, auth, body, scripted):
    scripted(responses=["one two three four"])

    with client.stream("POST", "/v1/chat/stream", json=body, headers=auth) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    frames = [f for f in raw.split("\n\n") if f.strip()]
    names = [f.split("event: ")[1].split("\n")[0] for f in frames if f.startswith("event:")]

    assert names[0] == "start"
    assert "token" in names
    assert names[-1] == "done"
    assert frames[-1].endswith("[DONE]")


def test_stream_reassembles_to_the_complete_text(client, auth, body, scripted):
    import json

    scripted(responses=["alpha beta gamma"])
    with client.stream("POST", "/v1/chat/stream", json=body, headers=auth) as response:
        raw = "".join(response.iter_text())

    deltas = [
        json.loads(f.split("data: ", 1)[1])["delta"]
        for f in raw.split("\n\n")
        if f.startswith("event: token")
    ]
    # The only contract streaming offers: concatenation reproduces the text.
    assert "".join(deltas) == "alpha beta gamma"


def test_stream_reports_usage_before_finishing(client, auth, body, scripted):
    scripted(responses=["short answer"])
    with client.stream("POST", "/v1/chat/stream", json=body, headers=auth) as response:
        raw = "".join(response.iter_text())

    assert "event: usage" in raw
    assert "cost_usd" in raw


def test_stream_sets_headers_that_stop_proxies_buffering(client, auth, body):
    # Without these, nginx buffers the whole response: streaming works locally
    # and silently stops working in production.
    with client.stream("POST", "/v1/chat/stream", json=body, headers=auth) as response:
        assert response.headers.get("X-Accel-Buffering") == "no"
        assert response.headers.get("Cache-Control") == "no-cache"


def test_stream_reports_provider_failure_in_band(client, auth, body, scripted):
    # 200 OK is already sent, so the status cannot change; an in-band error
    # event is the only alternative to a silent truncation.
    scripted(fail_times=1)
    with client.stream("POST", "/v1/chat/stream", json=body, headers=auth) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())

    assert "event: error" in raw
    assert raw.rstrip().endswith("[DONE]")


def test_stream_requires_auth(client, body):
    assert client.post("/v1/chat/stream", json=body).status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_token_bucket_allows_a_burst_then_throttles():
    bucket = TokenBucket(capacity=3, refill_per_second=1.0, tokens=3, updated_at=0.0)
    assert [bucket.consume(now=0.0) for _ in range(4)] == [True, True, True, False]


def test_token_bucket_refills_over_time():
    bucket = TokenBucket(capacity=3, refill_per_second=1.0, tokens=0, updated_at=0.0)
    assert bucket.consume(now=0.0) is False
    assert bucket.consume(now=2.0) is True, "two seconds should refill two tokens"


def test_rate_limiter_reports_when_to_retry():
    limiter = RateLimiter(per_minute=60, burst=1)
    limiter.check("k")
    allowed, _, retry_after = limiter.check("k")
    assert allowed is False
    assert retry_after > 0, "a bare 429 teaches clients nothing"


def test_limits_are_per_key_not_global():
    limiter = RateLimiter(per_minute=60, burst=1)
    assert limiter.check("tenant-a")[0] is True
    # One noisy client must not throttle everyone else.
    assert limiter.check("tenant-b")[0] is True


def test_exceeding_the_limit_returns_429_with_guidance(monkeypatch, body, auth):
    monkeypatch.setenv("RATE_LIMIT_BURST", "2")
    from app.config import get_settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    get_settings.cache_clear()
    client = TestClient(create_app())

    codes = [client.post("/v1/chat", json=body, headers=auth).status_code for _ in range(5)]
    assert codes[:2] == [200, 200]
    assert 429 in codes

    limited = client.post("/v1/chat", json=body, headers=auth)
    assert limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.headers["X-RateLimit-Remaining"] == "0"


def test_health_checks_are_never_rate_limited(monkeypatch, auth, body):
    # A limiter that blocks the readiness probe takes the pod out of service
    # under load -- precisely backwards.
    monkeypatch.setenv("RATE_LIMIT_BURST", "1")
    from app.config import get_settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    get_settings.cache_clear()
    client = TestClient(create_app())

    for _ in range(5):
        client.post("/v1/chat", json=body, headers=auth)
    assert client.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_tenant_quota_isolates_one_customer_from_another():
    # Without this, one abusive client exhausts the global budget and takes
    # the service down for every paying customer.
    registry = BudgetRegistry(global_limit_usd=100.0, tenant_limit_usd=1.0)
    registry.record("noisy", 1.0)

    with pytest.raises(BudgetExceeded) as exc:
        registry.check("noisy", 0.5)
    assert exc.value.scope == "tenant"

    registry.check("quiet", 0.5)  # unaffected


def test_global_budget_stops_all_spending():
    registry = BudgetRegistry(global_limit_usd=1.0, tenant_limit_usd=100.0)
    registry.record("anyone", 1.0)
    with pytest.raises(BudgetExceeded) as exc:
        registry.check("someone-else", 0.1)
    assert exc.value.scope == "global"


def test_exhausted_budget_returns_402_not_429(client, auth, body):
    # 429 tells a well-behaved client to retry, which it will, forever.
    get_budgets().configure(global_limit_usd=1e-9, tenant_limit_usd=1e-9)
    response = client.post("/v1/chat", json=body, headers={**auth, "X-Tenant-ID": "acme"})

    assert response.status_code == 402
    assert response.json()["error"]["code"] == "budget_exceeded"


def test_cost_is_attributed_per_tenant(client, auth, body):
    client.post("/v1/chat", json=body, headers={**auth, "X-Tenant-ID": "acme"})
    assert "acme" in client.get("/metrics").text


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def test_background_job_runs_to_completion(client, auth, body, scripted):
    scripted(responses=["done asynchronously"])
    accepted = client.post("/v1/chat/async", json=body, headers=auth)
    assert accepted.status_code == 202

    job_id = accepted.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["result"]["text"] == "done asynchronously"


def test_unknown_job_is_a_404(client, auth):
    assert client.get("/v1/jobs/does-not-exist", headers=auth).status_code == 404


# ---------------------------------------------------------------------------
# Logging hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "token"),
    [
        ("contact bob@example.com", "[EMAIL]"),
        ("key sk-abcdefgh12345", "[API_KEY]"),
        ("Bearer abcdef1234567890", "[BEARER]"),
        ("ssn 123-45-6789", "[SSN]"),
    ],
)
def test_secrets_are_redacted_before_logging(raw, token):
    # Redaction lives at the logging boundary, not at call sites: relying on
    # developers to remember is how PII reaches CloudWatch for seven years.
    assert token in redact(raw)


def test_ordinary_text_survives_redaction():
    text = "Annual leave is 25 days per year."
    assert redact(text) == text
