"""Why async is non-negotiable for LLM work -- measured, not asserted.

    python 03-llm-integration/examples/llmkit_ex_concurrent.py

FOR THE JAVA DEVELOPER

Your instinct is an ExecutorService with a thread pool. In Python that instinct
is wrong, and it is worth understanding exactly why.

The GIL (Global Interpreter Lock) allows only one thread to execute Python
bytecode at a time. Threads therefore buy you nothing for CPU-bound work -- but
they DO release the GIL during I/O, so threads are not useless here.

The real argument for asyncio over threads is cost and scale:

    Thread           ~8 MB of stack, kernel-scheduled, context switches cost
    asyncio Task     ~1 KB, scheduled in-process, switches are function calls

An LLM call is 2-20 SECONDS of doing nothing but waiting on a socket. Serving
1,000 concurrent LLM requests means 1,000 things waiting simultaneously. With
threads that is ~8 GB of stack and heavy scheduler pressure. With asyncio it is
a few MB.

    asyncio.gather      == CompletableFuture.allOf
    asyncio.Semaphore   == a bounded pool / Resilience4j Bulkhead
    asyncio.wait_for    == orTimeout / TimeLimiter
    return_exceptions   == allSettled semantics rather than fail-fast

THE ONE RULE: never call a blocking function inside async code. One
`time.sleep()` or one `requests.get()` freezes the entire event loop, and every
other in-flight request in the process stalls behind it. This is the single most
common async bug, and it is invisible until you are under load.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llmkit import FakeProvider, LLMClient, TransientError, user  # noqa: E402

LATENCY = 0.25  # seconds per simulated LLM call
QUESTIONS = [f"Summarize document #{i}" for i in range(12)]


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run_sequential() -> float:
    banner("1. SEQUENTIAL -- the obvious version")
    print(f"{len(QUESTIONS)} calls, {LATENCY}s each, one after another.\n")

    client = LLMClient(FakeProvider(latency=LATENCY))
    started = time.monotonic()
    for question in QUESTIONS:
        client.complete([user(question)])
    elapsed = time.monotonic() - started

    print(f"    elapsed: {elapsed:.2f}s")
    print(f"    ...of which {len(QUESTIONS) * LATENCY:.2f}s was spent waiting on a socket.")
    return elapsed


async def run_concurrent() -> float:
    banner("2. CONCURRENT -- asyncio.gather")
    print("The same calls, all in flight at once. Total time approaches the")
    print("SLOWEST single call rather than the sum of all of them.\n")

    client = LLMClient(FakeProvider(latency=LATENCY))
    started = time.monotonic()
    await asyncio.gather(*(client.acomplete([user(q)]) for q in QUESTIONS))
    elapsed = time.monotonic() - started

    print(f"    elapsed: {elapsed:.2f}s")
    return elapsed


async def run_bounded(limit: int = 4) -> float:
    banner(f"3. BOUNDED CONCURRENCY -- Semaphore({limit})")
    print("Unbounded gather() over 10,000 items will hit provider rate limits,")
    print("exhaust your connection pool, and blow up memory. A semaphore is the")
    print("bulkhead: at most N calls in flight, the rest queue politely.\n")

    client = LLMClient(FakeProvider(latency=LATENCY))
    semaphore = asyncio.Semaphore(limit)

    async def ask(question: str):
        # Acquire INSIDE the task, not around gather, so tasks queue rather
        # than all being created and then blocking.
        async with semaphore:
            return await client.acomplete([user(question)])

    started = time.monotonic()
    await asyncio.gather(*(ask(q) for q in QUESTIONS))
    elapsed = time.monotonic() - started

    waves = -(-len(QUESTIONS) // limit)  # ceiling division
    print(f"    elapsed: {elapsed:.2f}s  (~{waves} waves of {limit})")
    return elapsed


async def run_partial_failure() -> None:
    banner("4. PARTIAL FAILURE -- one bad call must not kill the batch")
    print("gather() is fail-fast by default: the first exception cancels the")
    print("rest and you lose work that had already succeeded. For batch jobs you")
    print("almost always want return_exceptions=True and per-item handling.\n")

    healthy = LLMClient(FakeProvider(responses=["ok"]))
    broken = LLMClient(FakeProvider(fail_times=99, failure=TransientError("503")), retry=None)

    async def maybe_fail(index: int):
        client = broken if index in (2, 5) else healthy
        return await client.acomplete([user(f"task {index}")])

    results = await asyncio.gather(*(maybe_fail(i) for i in range(8)), return_exceptions=True)

    succeeded = [r for r in results if not isinstance(r, BaseException)]
    failed = [r for r in results if isinstance(r, BaseException)]
    print(f"    succeeded: {len(succeeded)}/8")
    print(f"    failed   : {len(failed)}/8  ({', '.join(type(e).__name__ for e in failed)})")
    print("    -> the 6 successful results are still usable; retry only the 2 that failed.")


async def run_timeout() -> None:
    banner("5. PER-CALL TIMEOUT -- never wait forever")
    print("A hung request holds a worker until the heat death of the universe.")
    print("asyncio.wait_for bounds it. (3.9-compatible; 3.11+ has asyncio.timeout.)\n")

    slow = LLMClient(FakeProvider(latency=2.0))
    started = time.monotonic()
    try:
        await asyncio.wait_for(slow.acomplete([user("slow one")]), timeout=0.3)
    except asyncio.TimeoutError:
        print(f"    timed out after {time.monotonic() - started:.2f}s (limit was 0.30s)")
        print("    -> the task is cancelled, the worker is freed")


async def main() -> None:
    print(__doc__)

    sequential = run_sequential()
    concurrent = await run_concurrent()
    await run_bounded()
    await run_partial_failure()
    await run_timeout()

    banner("RESULT")
    speedup = sequential / concurrent if concurrent else 0.0
    print(f"    sequential : {sequential:.2f}s")
    print(f"    concurrent : {concurrent:.2f}s")
    print(f"    speedup    : {speedup:.1f}x")
    print(
        "\n    Same machine. Same number of calls. No extra CPU. The work was\n"
        "    always waiting, and waiting is free if you do it concurrently.\n"
        "\n    This is why every LLM endpoint you write should be `async def`."
    )


if __name__ == "__main__":
    asyncio.run(main())
