"""The ReAct loop, written out.

ReAct = Reasoning + Acting (Yao et al., 2022). The model alternates between
thinking and calling tools:

    Thought: I need yesterday's order count, which means querying the database.
    Action: sql_query
    Action Input: {"query": "SELECT count(*) FROM orders WHERE ..."}
    Observation: 1,284
    Thought: That answers the question.
    Final Answer: There were 1,284 orders yesterday.

That is the entire idea. The loop below is ~80 lines; everything AROUND it --
the termination guarantees -- is what makes it usable in production.

WHY A TEXT PROTOCOL AT ALL?

Modern models have native tool calling (see `function_calling.py`), and you
should prefer it. ReAct still earns its place:

- it works with ANY model, including local ones with no tool-calling support
- the reasoning is explicit text you can read, log, and evaluate
- implementing it once teaches you what native tool calling is doing for you

Its weakness is the parser: you are extracting structure from prose, and
models do not always cooperate. The parser here is deliberately forgiving.

THE FIVE WAYS AN AGENT LOOP MUST BE ABLE TO STOP

A loop whose exit condition is "the model says it's done" is not a loop, it is
a hope. Every one of these is a real failure observed in production:

    1. max_iterations    the model never concludes
    2. budget_exceeded   spend ceiling reached mid-run
    3. deadline_exceeded wall-clock limit reached
    4. loop_detected     same tool, same arguments, repeatedly
    5. no_progress       tool keeps erroring, nothing is changing

Hitting any of these is NOT success. `AgentResult.stop_reason` records which,
and `succeeded` is only true for `final_answer` -- because an agent that
silently returns its best guess after exhausting the iteration cap looks
exactly like one that answered correctly.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .tools import ToolRegistry
from .types import (
    AgentResult,
    LLMResponse,
    Message,
    Step,
    Timer,
    ToolCall,
    Usage,
    assistant,
    system,
    user,
)

logger = logging.getLogger(__name__)

REACT_PROMPT = """You are a careful assistant that solves problems step by step using tools.

Available tools:
{tools}

Use EXACTLY this format:

Thought: <your reasoning about what to do next>
Action: <one tool name from the list above>
Action Input: <a JSON object of arguments>

You will then be shown:

Observation: <the tool's output>

Repeat Thought/Action/Action Input as many times as needed. When you can answer,
use exactly:

Thought: <why you can now answer>
Final Answer: <your answer>

Rules:
- Output ONE Thought and ONE Action per turn, then stop and wait.
- Action Input must be valid JSON.
- If a tool returns an error, read it and try a DIFFERENT approach.
- If you cannot answer with the tools available, say so in the Final Answer."""


_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_INPUT_RE = re.compile(
    r"Action\s*Input\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)
_THOUGHT_RE = re.compile(
    r"Thought\s*:\s*(.+?)(?=\n\s*(?:Action|Final Answer)\s*:|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedStep:
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[dict] = None
    final_answer: Optional[str] = None
    parse_error: Optional[str] = None


def parse_react(text: str) -> ParsedStep:
    """Extract Thought / Action / Action Input / Final Answer from prose.

    Deliberately forgiving, because models improvise: they wrap JSON in
    markdown fences, use single quotes, add trailing commas, or emit a bare
    string where an object belongs. Every one of those is recoverable, and
    failing the run over punctuation would be absurd.

    >>> parse_react("Thought: done\\nFinal Answer: 42").final_answer
    '42'
    """
    # Check Final Answer FIRST. A model often emits a last Thought alongside
    # it, and treating that as another action would loop forever.
    final = _FINAL_RE.search(text)
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    if final:
        return ParsedStep(thought=thought, final_answer=final.group(1).strip())

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return ParsedStep(thought=thought, parse_error="no Action or Final Answer found")

    action = action_match.group(1).strip().strip("`\"'")
    input_match = _INPUT_RE.search(text)
    if not input_match:
        # A no-argument tool is legitimate.
        return ParsedStep(thought=thought, action=action, action_input={})

    raw = input_match.group(1).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", raw)
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return ParsedStep(
                thought=thought,
                action=action,
                parse_error=f"Action Input was not valid JSON: {raw[:120]}",
            )

    if not isinstance(parsed, dict):
        return ParsedStep(
            thought=thought, action=action, parse_error="Action Input must be a JSON object"
        )
    return ParsedStep(thought=thought, action=action, action_input=parsed)


@dataclass
class ReActAgent:
    """A ReAct agent with enforced termination.

    Args:
        provider: anything with `.complete(messages) -> LLMResponse`.
        registry: the tools it may use.
        max_iterations: hard cap on loop turns.
        max_cost_usd: spend ceiling, checked every turn.
        deadline_seconds: wall-clock ceiling.
        loop_window: how many recent actions to compare for repetition.
        cost_per_1k_tokens: for the running cost estimate.
        on_step: callback for streaming progress to a UI or log.
    """

    provider: Any
    registry: ToolRegistry
    max_iterations: int = 8
    max_cost_usd: float = 1.0
    deadline_seconds: float = 120.0
    loop_window: int = 3
    cost_per_1k_tokens: float = 0.0006
    on_step: Optional[Callable[[Step], None]] = None
    system_prompt: str = REACT_PROMPT

    def _cost(self, usage: Usage) -> float:
        return (usage.total_tokens / 1000.0) * self.cost_per_1k_tokens

    def _finish(
        self,
        answer: str,
        steps: list[Step],
        usage: Usage,
        reason: str,
        elapsed: float,
    ) -> AgentResult:
        if reason != "final_answer":
            logger.warning("agent stopped early: %s after %d steps", reason, len(steps))
        return AgentResult(
            answer=answer,
            steps=steps,
            usage=usage,
            cost_usd=self._cost(usage),
            stop_reason=reason,
            elapsed_ms=elapsed,
        )

    def run(self, question: str, *, context: Optional[Sequence[Message]] = None) -> AgentResult:
        started = time.monotonic()
        messages: list[Message] = [
            system(self.system_prompt.format(tools=self.registry.describe())),
            *(context or []),
            user(question),
        ]

        steps: list[Step] = []
        usage = Usage()
        recent_signatures: list[str] = []
        consecutive_errors = 0

        for index in range(self.max_iterations):
            elapsed = time.monotonic() - started
            if elapsed > self.deadline_seconds:
                return self._finish(
                    self._best_effort(steps), steps, usage, "deadline_exceeded", elapsed * 1000
                )
            if self._cost(usage) > self.max_cost_usd:
                # Checked BEFORE the call, so the ceiling is a ceiling.
                return self._finish(
                    self._best_effort(steps), steps, usage, "budget_exceeded", elapsed * 1000
                )

            with Timer() as turn_timer:
                response: LLMResponse = self.provider.complete(messages)
            usage = usage + response.usage
            parsed = parse_react(response.text)

            step = Step(
                index=index,
                thought=parsed.thought,
                usage=response.usage,
                elapsed_ms=turn_timer.elapsed_ms,
            )

            if parsed.final_answer is not None:
                steps.append(step)
                if self.on_step:
                    self.on_step(step)
                return self._finish(
                    parsed.final_answer,
                    steps,
                    usage,
                    "final_answer",
                    (time.monotonic() - started) * 1000,
                )

            if parsed.parse_error or not parsed.action:
                # Tell the model precisely what was wrong with its output.
                # Vague nagging ("invalid format") produces another invalid
                # response; quoting the required format usually fixes it.
                consecutive_errors += 1
                steps.append(step)
                if consecutive_errors >= 3:
                    return self._finish(
                        self._best_effort(steps),
                        steps,
                        usage,
                        "no_progress",
                        (time.monotonic() - started) * 1000,
                    )
                messages.extend(
                    [
                        assistant(response.text),
                        user(
                            f"Your output could not be parsed: {parsed.parse_error}. "
                            "Reply using exactly:\nThought: ...\nAction: <tool>\n"
                            'Action Input: {"arg": "value"}\nor\nFinal Answer: ...'
                        ),
                    ]
                )
                continue

            call = ToolCall(
                id=f"call_{index}", name=parsed.action, arguments=parsed.action_input or {}
            )
            step.tool_call = call

            signature = call.signature()
            if recent_signatures.count(signature) >= 2:
                # The same call twice already produced the same observation.
                # A third will too; the model is stuck, not thinking.
                steps.append(step)
                return self._finish(
                    self._best_effort(steps),
                    steps,
                    usage,
                    "loop_detected",
                    (time.monotonic() - started) * 1000,
                )
            recent_signatures.append(signature)
            recent_signatures[:] = recent_signatures[-self.loop_window :]

            result = self.registry.execute(call)
            step.result = result
            steps.append(step)
            if self.on_step:
                self.on_step(step)

            consecutive_errors = consecutive_errors + 1 if not result.ok else 0
            if consecutive_errors >= 3:
                return self._finish(
                    self._best_effort(steps),
                    steps,
                    usage,
                    "no_progress",
                    (time.monotonic() - started) * 1000,
                )

            messages.extend(
                [
                    assistant(response.text),
                    user(f"Observation: {result.observation}"),
                ]
            )

        return self._finish(
            self._best_effort(steps),
            steps,
            usage,
            "max_iterations",
            (time.monotonic() - started) * 1000,
        )

    @staticmethod
    def _best_effort(steps: Sequence[Step]) -> str:
        """Degrade gracefully instead of returning nothing.

        The run failed, and `stop_reason` says so -- but the last successful
        observation is usually more useful to a caller than an empty string,
        and a UI can present it as partial progress.
        """
        for step in reversed(steps):
            if step.result and step.result.ok:
                return (
                    "I could not complete this fully. Based on what I found: "
                    f"{step.result.content[:500]}"
                )
        return "I was unable to complete this task."
