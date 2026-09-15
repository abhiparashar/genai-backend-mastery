"""Getting typed, validated data out of a text-generating model.

THE PROBLEM

An LLM returns a string. Your application needs a `Invoice(total=Decimal, ...)`.
Everything between those two facts is where production LLM code actually breaks.

The naive version -- `json.loads(response.text)` -- fails constantly, and always
in production rather than in your demo:

    Here's the JSON you requested:          <- prose preamble
    ```json                                  <- markdown fence
    {"total": 42.50,}                        <- trailing comma
    ```
    Let me know if you need anything else!   <- prose postamble

and occasionally the response is simply truncated mid-object because it hit
`max_tokens` (check `finish_reason == "length"` -- this is why `LLMResponse`
exposes `.truncated`).

THE FIX, IN FOUR LAYERS, CHEAPEST FIRST

1. **Ask correctly.** Use the provider's JSON mode / structured-output feature
   when available, and put the schema in the prompt. Most failures never happen.
2. **Extract.** Strip fences and prose; find the outermost balanced JSON object.
3. **Repair.** Fix the small, mechanical mistakes models actually make
   (trailing commas, single quotes, Python literals).
4. **Re-ask, bounded.** If validation still fails, send the validation errors
   back to the model and let it correct itself. Bounded, because an unbounded
   repair loop is an unbounded bill.

Layer 4 is the one people skip, and it is the one that takes a pipeline from
~95% to ~99.9% success. Feeding the model its own Pydantic error message works
remarkably well -- it is precise, machine-generated feedback.

The Java analogue is Jackson deserialization plus Bean Validation, except the
producer is a language model that has merely been *asked* to honour the schema.
Trust nothing; validate everything.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .types import LLMResponse, Message, assistant, user

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(Exception):
    """Raised when we could not obtain a valid object within the retry budget."""

    def __init__(self, message: str, *, raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Layer 2: extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Optional[str]:
    """Pull the most likely JSON payload out of a model response.

    Strategy, in order: fenced code block, then the outermost balanced {...} or
    [...] span. Balanced scanning beats a regex because JSON nests, and it is
    string-aware so a brace inside a string value does not break the count.

    >>> extract_json('Sure!\\n```json\\n{"a": 1}\\n```\\nHope that helps')
    '{"a": 1}'
    >>> extract_json('no json here') is None
    True
    """
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Layer 3: repair
# ---------------------------------------------------------------------------


def repair_json(text: str) -> str:
    """Fix the mechanical mistakes models actually make.

    Deliberately conservative. Aggressive "smart" repair invents data, which is
    far worse than a clean failure -- you want a loud error, not a silently
    wrong invoice total.

    >>> repair_json('{"a": 1,}')
    '{"a": 1}'
    >>> json.loads(repair_json("{'a': True, 'b': None}"))
    {'a': True, 'b': None}
    """
    out = text.strip()

    # Python literals leaking through from a model that "thinks" in Python.
    out = re.sub(r"\bTrue\b", "true", out)
    out = re.sub(r"\bFalse\b", "false", out)
    out = re.sub(r"\bNone\b", "null", out)

    # Single-quoted keys/values -> double-quoted. Only when the text contains no
    # double quotes at all, so we cannot corrupt a legitimate apostrophe inside
    # a properly quoted string.
    if "'" in out and '"' not in out:
        out = out.replace("'", '"')

    # Trailing commas before a closing brace/bracket.
    return re.sub(r",(\s*[}\]])", r"\1", out)


def parse_json_loose(text: str) -> Any:
    """Extract + repair + parse. Raises `json.JSONDecodeError` on failure."""
    candidate = extract_json(text)
    if candidate is None:
        candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(repair_json(candidate))


# ---------------------------------------------------------------------------
# Layer 1: asking correctly
# ---------------------------------------------------------------------------


def schema_instruction(model: type[BaseModel]) -> str:
    """Prompt fragment describing the required shape.

    Sends the real JSON Schema rather than a hand-written description, so the
    prompt can never drift out of sync with the model class -- the schema is
    generated from the same source of truth that will validate the response.
    """
    schema = json.dumps(model.model_json_schema(), indent=2)
    return (
        "Respond with a single JSON object matching this JSON Schema.\n"
        "Output raw JSON only: no prose, no markdown fences, no explanation.\n\n"
        f"{schema}"
    )


def build_messages(
    prompt: str,
    model: type[BaseModel],
    *,
    system_prompt: Optional[str] = None,
) -> list[Message]:
    from .types import system as system_message

    messages: list[Message] = []
    if system_prompt:
        messages.append(system_message(system_prompt))
    messages.append(user(f"{prompt}\n\n{schema_instruction(model)}"))
    return messages


# ---------------------------------------------------------------------------
# Layer 4: bounded re-ask
# ---------------------------------------------------------------------------


def _correction_prompt(raw: str, error: str) -> Message:
    return user(
        "That response was not valid according to the schema.\n\n"
        f"Validation errors:\n{error}\n\n"
        "Return ONLY corrected raw JSON. Do not apologise or explain."
    )


def generate_structured(
    complete: Callable[[Sequence[Message]], LLMResponse],
    prompt: str,
    schema: type[M],
    *,
    system_prompt: Optional[str] = None,
    max_attempts: int = 3,
) -> M:
    """Generate a validated instance of `schema`.

    Args:
        complete: anything that maps messages -> LLMResponse. Usually
            `client.complete`; a bare provider works too, which keeps this
            function trivially testable with FakeProvider.
        max_attempts: total tries including the first. Each retry costs another
            full call, so keep this small -- 3 is the sweet spot. If you need 5,
            your prompt or your schema is the problem.

    Raises:
        StructuredOutputError: exhausted attempts without a valid object.
    """
    messages = build_messages(prompt, schema, system_prompt=system_prompt)
    last_raw = ""
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        response = complete(messages)
        last_raw = response.text

        if response.truncated:
            # A truncated response is never valid JSON, and re-asking without
            # raising max_tokens will truncate again. Fail loudly with the real
            # cause instead of burning retries on a parse error.
            raise StructuredOutputError(
                "response hit max_tokens and was truncated; raise max_tokens "
                "or reduce the requested output size",
                raw=last_raw,
                attempts=attempt,
            )

        try:
            data = parse_json_loose(response.text)
        except json.JSONDecodeError as exc:
            last_error = f"response was not valid JSON: {exc}"
        else:
            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                last_error = str(exc)

        logger.info("structured output attempt %d/%d failed: %s", attempt, max_attempts, last_error)

        if attempt < max_attempts:
            # Thread the failed attempt AND the error back into the conversation
            # so the model sees exactly what it got wrong.
            messages = [
                *messages,
                assistant(response.text),
                _correction_prompt(response.text, last_error),
            ]

    raise StructuredOutputError(
        f"could not obtain valid {schema.__name__} after {max_attempts} attempts: {last_error}",
        raw=last_raw,
        attempts=max_attempts,
    )


async def generate_structured_async(
    acomplete: Callable[[Sequence[Message]], Any],
    prompt: str,
    schema: type[M],
    *,
    system_prompt: Optional[str] = None,
    max_attempts: int = 3,
) -> M:
    """Async twin of `generate_structured`."""
    messages = build_messages(prompt, schema, system_prompt=system_prompt)
    last_raw = ""
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        response: LLMResponse = await acomplete(messages)
        last_raw = response.text

        if response.truncated:
            raise StructuredOutputError(
                "response hit max_tokens and was truncated",
                raw=last_raw,
                attempts=attempt,
            )

        try:
            data = parse_json_loose(response.text)
        except json.JSONDecodeError as exc:
            last_error = f"response was not valid JSON: {exc}"
        else:
            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                last_error = str(exc)

        if attempt < max_attempts:
            messages = [
                *messages,
                assistant(response.text),
                _correction_prompt(response.text, last_error),
            ]

    raise StructuredOutputError(
        f"could not obtain valid {schema.__name__} after {max_attempts} attempts: {last_error}",
        raw=last_raw,
        attempts=max_attempts,
    )
