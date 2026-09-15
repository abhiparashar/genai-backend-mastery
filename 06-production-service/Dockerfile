# Multi-stage build. The builder carries compilers and build headers; the
# runtime image must not. Shipping a toolchain to production enlarges both the
# image and the attack surface for no benefit.

# ---------------------------------------------------------------------------
# Stage 1: build dependencies into a wheel cache
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Pin the digest in a real deployment:
#   FROM python:3.11-slim@sha256:<digest>
# A floating tag means your "reproducible" build silently changes base image
# under you, which is a production incident waiting for a slow Tuesday.

WORKDIR /build

# Copy ONLY the requirements first. Docker caches layers by content, so this
# layer is reused on every build where dependencies did not change - turning a
# 90-second rebuild into 3 seconds. Copying the source first destroys that.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Run as a non-root user. Container escapes are rarer than they used to be,
# but root inside a container is root on the host kernel if one is found, and
# this costs two lines.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=app:app app ./app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    LOG_FORMAT=json
# PYTHONUNBUFFERED matters more than it looks: without it Python buffers
# stdout, so your logs arrive in chunks - or not at all if the process is
# killed - and debugging a crash becomes guesswork.

USER app
EXPOSE 8000

# Hits liveness, which by design touches no dependencies. A healthcheck that
# calls /readyz would restart the container whenever a dependency blips.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=2).status==200 else 1)"

# One worker per container, scaled by replica count rather than by workers.
# The orchestrator already does process supervision, load balancing and
# restarts; running 4 workers inside one container hides 3 of them from it.
#
# For CPU-bound workloads you would use gunicorn with uvicorn workers. This
# service is I/O-bound (it waits on an LLM), so async concurrency inside a
# single process is exactly the right model.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
# --no-access-log because RequestLoggingMiddleware already emits one
# structured JSON line per request; uvicorn's plaintext duplicate is noise.
