"""Chat endpoints: synchronous, streaming (SSE), and background jobs.

WHY STREAMING MATTERS

An LLM takes 2-20 seconds to produce a full answer but emits the first token
in a few hundred milliseconds. Streaming does not make generation faster --
it makes the WAIT visible, and perceived latency drops by an order of
magnitude. It is the single highest-impact UX change in an LLM product, and
it costs one endpoint.

SSE VS WEBSOCKET

Server-Sent Events, not WebSocket, because the traffic is one-directional:
the server streams, the client listens. SSE is plain HTTP, so it keeps
proxies, load balancers, auth headers and HTTP/2 multiplexing. WebSocket
gives you bidirectionality you do not need in exchange for an upgrade
handshake and infrastructure that often mishandles it.

THE THREE THINGS PEOPLE GET WRONG WITH SSE

1. **Framing.** Each event is `data: <payload>\\n\\n`. The blank line is the
   delimiter. Omit it and the client buffers forever.
2. **Disconnects.** If the client vanishes mid-stream you must stop calling
   the provider. Otherwise you pay for tokens nobody receives.
3. **Errors after the first byte.** You have already sent 200 OK, so you
   cannot change the status. Errors must be sent as an in-band `error` event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import Settings, get_settings
from ..costs import BudgetExceeded, BudgetRegistry
from ..deps import Principal, get_budgets, get_provider, require_api_key
from ..llm import LLMError, Message, Usage, estimate_cost, estimate_tokens
from ..observability import METRICS, correlation_id_var, log_with
from ..schemas import ChatRequest, ChatResponse, JobAccepted, JobStatus, UsageOut
from ..security import ValidationProblem, check_prompt_size, check_repetition, detect_injection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])

# In-memory job store. Fine for a reference implementation; a real deployment
# needs Redis or a database, because this dies with the process and is not
# shared across replicas. Said plainly rather than left for you to discover.
_JOBS: dict[str, JobStatus] = {}


def _to_messages(request: ChatRequest) -> list:
    return [Message(role=m.role, content=m.content) for m in request.messages]


def _validate(request: ChatRequest, settings: Settings) -> None:
    """Edge checks that run before any spend."""
    try:
        check_prompt_size(request.total_chars(), settings.max_prompt_chars)
        for message in request.messages:
            check_repetition(message.content)
    except ValidationProblem as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_request",
                "message": str(exc),
                "correlation_id": correlation_id_var.get(),
            },
        ) from exc

    # Advisory only -- logged, never blocking. Blocking on a heuristic means
    # rejecting legitimate prompts that merely discuss prompt injection.
    last = request.messages[-1].content
    categories = detect_injection(last)
    if categories:
        log_with(
            logger, logging.WARNING, "possible prompt injection in request",
            categories=categories,
        )


def _project_cost(request: ChatRequest, model: str, settings: Settings) -> float:
    prompt_tokens = estimate_tokens(" ".join(m.content for m in request.messages))
    expected_output = request.max_tokens or min(1024, settings.max_output_tokens)
    return estimate_cost(Usage(prompt_tokens, expected_output), model)


def _guard_budget(budgets: BudgetRegistry, tenant: str, projected: float) -> None:
    try:
        budgets.check(tenant, projected)
    except BudgetExceeded as exc:
        # 402, not 429: retrying will not help, and telling a client to retry
        # an exhausted budget produces an infinite polite loop.
        raise HTTPException(
            status_code=402,
            detail={
                "code": "budget_exceeded",
                "message": str(exc),
                "correlation_id": correlation_id_var.get(),
            },
        ) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    principal: Principal = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    provider: Any = Depends(get_provider),
    budgets: BudgetRegistry = Depends(get_budgets),
) -> ChatResponse:
    """Single completion."""
    _validate(request, settings)
    model = request.model or settings.llm_model
    _guard_budget(budgets, principal.tenant, _project_cost(request, model, settings))

    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            provider.acomplete(
                _to_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            timeout=settings.llm_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "code": "upstream_error",
                "message": f"provider timed out after {settings.llm_timeout_seconds}s",
                "correlation_id": correlation_id_var.get(),
            },
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "upstream_error",
                "message": "the model provider failed",
                "correlation_id": correlation_id_var.get(),
            },
        ) from exc

    duration_ms = (time.monotonic() - started) * 1000.0
    cost = estimate_cost(result.usage, result.model)
    budgets.record(principal.tenant, cost)
    METRICS.record_llm(
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=cost,
        duration_ms=duration_ms,
        tenant=principal.tenant,
    )
    log_with(
        logger, logging.INFO, "chat completed",
        model=result.model, tokens=result.usage.total_tokens,
        cost_usd=round(cost, 6), latency_ms=round(duration_ms),
    )

    return ChatResponse(
        text=result.text,
        model=result.model,
        finish_reason=result.finish_reason,
        usage=UsageOut(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        cost_usd=round(cost, 8),
        latency_ms=round(duration_ms, 1),
        cached=result.cached,
        correlation_id=correlation_id_var.get(),
    )


def _sse(event: str, data: Any) -> str:
    """Format one SSE frame.

    The trailing blank line is the frame delimiter. Without it the client
    buffers indefinitely and the stream appears to hang -- the single most
    common SSE bug.
    """
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    provider: Any = Depends(get_provider),
    budgets: BudgetRegistry = Depends(get_budgets),
) -> StreamingResponse:
    """Stream tokens as Server-Sent Events.

    Named events (`start`/`token`/`usage`/`error`/`done`) rather than anonymous
    frames: the client dispatches on the event name instead of sniffing JSON
    keys to tell a heartbeat from a token.
    """
    _validate(request, settings)
    model = request.model or settings.llm_model
    _guard_budget(budgets, principal.tenant, _project_cost(request, model, settings))
    correlation_id = correlation_id_var.get()

    async def generate():
        started = time.monotonic()
        stream_id = uuid.uuid4().hex[:12]
        text_parts: list = []

        yield _sse("start", {"id": stream_id, "model": model})
        try:
            async for chunk in provider.astream(
                _to_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                # Stop paying for tokens nobody will receive.
                if await http_request.is_disconnected():
                    logger.info("client disconnected; aborting stream %s", stream_id)
                    return
                text_parts.append(chunk)
                yield _sse("token", {"delta": chunk})

        except LLMError as exc:
            # 200 OK was already sent, so the status cannot change. The error
            # must travel in-band or the client sees a silent truncation.
            logger.warning("stream failed mid-flight: %s", exc)
            yield _sse("error", {"code": "upstream_error", "message": "the model provider failed"})
            yield _sse("done", "[DONE]")
            return

        text = "".join(text_parts)
        usage = Usage(
            input_tokens=estimate_tokens(" ".join(m.content for m in request.messages)),
            output_tokens=estimate_tokens(text),
        )
        cost = estimate_cost(usage, model)
        duration_ms = (time.monotonic() - started) * 1000.0

        budgets.record(principal.tenant, cost)
        METRICS.record_llm(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            tenant=principal.tenant,
        )

        yield _sse(
            "usage",
            {
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "cost_usd": round(cost, 8),
                "latency_ms": round(duration_ms, 1),
            },
        )
        yield _sse("done", "[DONE]")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            # Without these, nginx buffers the whole response and streaming
            # silently stops working in production while it worked locally.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Correlation-ID": correlation_id,
        },
    )


async def _run_job(job_id: str, request: ChatRequest, provider: Any, settings: Settings,
                   budgets: BudgetRegistry, tenant: str) -> None:
    _JOBS[job_id] = JobStatus(job_id=job_id, status="running")
    try:
        result = await provider.acomplete(_to_messages(request), temperature=request.temperature)
        cost = estimate_cost(result.usage, result.model)
        budgets.record(tenant, cost)
        _JOBS[job_id] = JobStatus(
            job_id=job_id,
            status="succeeded",
            result=ChatResponse(
                text=result.text,
                model=result.model,
                finish_reason=result.finish_reason,
                usage=UsageOut(
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    total_tokens=result.usage.total_tokens,
                ),
                cost_usd=round(cost, 8),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - recorded as job state
        logger.exception("job %s failed", job_id)
        _JOBS[job_id] = JobStatus(
            job_id=job_id,
            status="failed",
            error={"code": "upstream_error", "message": str(exc)[:200]},
        )


@router.post("/chat/async", response_model=JobAccepted, status_code=202)
async def chat_async(
    request: ChatRequest,
    background: BackgroundTasks,
    principal: Principal = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    provider: Any = Depends(get_provider),
    budgets: BudgetRegistry = Depends(get_budgets),
) -> JobAccepted:
    """Submit work and poll for it.

    For anything that may outlive an HTTP timeout: long agent runs, batch
    jobs, document processing. 202 Accepted plus a poll URL is the honest
    answer to "this will take two minutes".

    BackgroundTasks runs in THIS process, so work is lost on restart and does
    not survive a deploy. Real queues (Celery, RQ, SQS) exist for that; this
    shows the API shape, which is the part that transfers.
    """
    _validate(request, settings)
    job_id = uuid.uuid4().hex[:16]
    _JOBS[job_id] = JobStatus(job_id=job_id, status="queued")
    background.add_task(_run_job, job_id, request, provider, settings, budgets, principal.tenant)
    return JobAccepted(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, principal: Principal = Depends(require_api_key)) -> JobStatus:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "invalid_request",
                "message": f"no such job: {job_id}",
                "correlation_id": correlation_id_var.get(),
            },
        )
    return job
