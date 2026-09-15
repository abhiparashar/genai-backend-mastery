"""Watch retry, fallback, and the circuit breaker actually fire.

    python 03-llm-integration/examples/llmkit_ex_resilience.py

Runs entirely offline. Nothing here is simulated output -- every line printed is
produced by the real code paths in llmkit, driven by a FakeProvider scripted to
fail in specific ways.

This is the example to read if you only read one. Resilience is the difference
between "I called the OpenAI API" and "I run an LLM service".
"""

from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llmkit import (  # noqa: E402
    CircuitBreaker,
    CircuitOpenError,
    FakeProvider,
    LLMClient,
    RateLimitError,
    RetryPolicy,
    TransientError,
    user,
)
from llmkit.circuit import manual_clock  # noqa: E402

# INFO so llmkit's own structured call log is visible -- that one line per call
# is what you would ship to your log aggregator.
logging.basicConfig(level=logging.INFO, format="    %(levelname)-7s %(message)s")

FAST = RetryPolicy(max_attempts=4, base_delay=0.01, jitter=True)


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def scenario_retry() -> None:
    banner("1. RETRY -- two 429s, then success")
    print("A rate limit is transient. Backing off and retrying is correct;")
    print("failing the user's request is not.\n")

    provider = FakeProvider(
        responses=["Paris"], fail_times=2, failure=RateLimitError("429", retry_after=0)
    )
    client = LLMClient(provider, retry=FAST, default_model="gpt-4o-mini")

    response = client.complete([user("Capital of France?")])
    print(f"\n    -> answer   : {response.text}")
    print(f"    -> attempts : {provider.calls} (2 failed, 1 succeeded)")


def scenario_fatal_not_retried() -> None:
    banner("2. NO RETRY -- a 401 is retried zero times")
    print("A bad API key will not become valid in 800ms. Retrying it wastes")
    print("latency and rate-limit budget for a guaranteed failure.\n")

    from llmkit import AuthError

    provider = FakeProvider(fail_times=99, failure=AuthError("invalid api key"))
    client = LLMClient(provider, retry=FAST)

    try:
        client.complete([user("hi")])
    except AuthError as exc:
        print(f"\n    -> raised   : {type(exc).__name__}: {exc}")
        print(f"    -> attempts : {provider.calls} (correct: fatal errors fail fast)")


def scenario_fallback() -> None:
    banner("3. FALLBACK -- primary provider is down, secondary answers")
    print("Provider outages are not rare. A second provider keeps you serving,")
    print("at the cost of different output quality -- so it is logged loudly.\n")

    primary = FakeProvider(fail_times=99, failure=TransientError("503 Service Unavailable"))
    secondary = FakeProvider(responses=["Answer from the backup provider"])
    client = LLMClient(
        primary, fallbacks=[secondary], retry=RetryPolicy(max_attempts=2, base_delay=0.01)
    )

    response = client.complete([user("hi")])
    print(f"\n    -> answer          : {response.text}")
    print(f"    -> primary calls   : {primary.calls} (all failed)")
    print(f"    -> secondary calls : {secondary.calls}")


def scenario_circuit_breaker() -> None:
    banner("4. CIRCUIT BREAKER -- stop hammering a provider that is down")
    print("Retry handles a blip but makes an outage worse: every request now")
    print("costs 3 attempts. After N consecutive failures the breaker opens and")
    print("requests fail instantly, shedding load and freeing your workers.\n")

    clock = manual_clock()
    provider = FakeProvider(fail_times=99, failure=TransientError("503"))
    breaker = CircuitBreaker(
        failure_threshold=3, recovery_timeout=30, success_threshold=2, clock=clock
    )
    client = LLMClient(provider, retry=RetryPolicy(max_attempts=1), breaker=breaker)

    for i in range(3):
        try:
            client.complete([user("hi")])
        except TransientError:
            print(f"    request {i + 1}: failed   (breaker={breaker.state.value})")

    print(f"\n    breaker is now {breaker.state.value.upper()}\n")

    for i in range(3):
        try:
            client.complete([user("hi")])
        except CircuitOpenError as exc:
            print(
                f"    request {i + 4}: rejected instantly, "
                f"retry in {exc.retry_after:.0f}s (provider never called)"
            )

    print(f"\n    provider calls so far: {provider.calls} (not 6 -- the breaker absorbed 3)")

    print("\n    ...30 seconds pass, provider recovers...\n")
    clock.advance(31)
    provider.fail_times = 0  # provider is healthy again
    print(f"    breaker moved to {breaker.state.value.upper()} (probing with limited traffic)")

    client.complete([user("hi")])
    print(f"    probe 1 succeeded (breaker={breaker.state.value})")
    client.complete([user("hi")])
    print(f"    probe 2 succeeded (breaker={breaker.state.value.upper()}) -- fully recovered")


def scenario_budget() -> None:
    banner("5. BUDGET GUARD -- refuse to spend past a ceiling")
    print("The 2am failure mode: an agent loop makes 40,000 calls and turns a")
    print("$12/day service into a $4,000 invoice. The guard runs BEFORE the call.\n")

    from llmkit import BudgetExceededError, BudgetGuard

    guard = BudgetGuard(limit_usd=0.05)
    client = LLMClient(FakeProvider(), budget=guard, default_model="gpt-4o")

    sent = 0
    try:
        for _ in range(100):
            client.complete([user("write an essay " * 200)], model="gpt-4o")
            sent += 1
    except BudgetExceededError as exc:
        print(f"\n    -> stopped after {sent} calls")
        print(f"    -> spent  ${exc.spent_usd:.4f} of ${exc.limit_usd:.4f} limit")
        print("    -> the blocked call never reached the provider")


def main() -> None:
    print(__doc__)
    scenario_retry()
    scenario_fatal_not_retried()
    scenario_fallback()
    scenario_circuit_breaker()
    scenario_budget()

    banner("TAKEAWAY")
    print(
        "Retry what can succeed. Fail fast on what cannot. Stop calling a dead\n"
        "provider. Always have a second option. Never let spend run unbounded.\n"
        "\nAll five are ~200 lines of code, and all five are the difference between\n"
        "a demo and a service."
    )


if __name__ == "__main__":
    main()
