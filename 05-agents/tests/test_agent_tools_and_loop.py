"""Tool schema derivation, sandboxing, the ReAct loop, and memory.

The loop tests matter most: they assert the TERMINATION guarantees. An agent
that cannot be forced to stop is not shippable, and "the model usually
finishes" is not a guarantee.
"""

from __future__ import annotations

import itertools
import sqlite3
from typing import Literal, Optional

import pytest
from agentkit.memory import (
    BufferMemory,
    ConversationBudget,
    SummaryMemory,
    TokenWindowMemory,
    VectorMemory,
    WindowMemory,
    estimate_tokens,
)
from agentkit.react import ReActAgent, parse_react
from agentkit.tools import (
    ToolRegistry,
    calculator,
    make_read_file_tool,
    make_search_tool,
    make_sql_tool,
    safe_eval,
    tool,
)
from agentkit.types import FakeProvider, Message, ToolCall, system, user

# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------


@tool
def sample(
    city: str, days: int = 3, units: Literal["c", "f"] = "c", tags: Optional[list] = None
) -> str:
    """Get a forecast.

    Args:
        city: City name.
        days: Days ahead.
    """
    return f"{city}/{days}/{units}/{tags}"


def test_schema_is_derived_from_type_hints():
    properties = sample.parameters["properties"]
    assert properties["city"]["type"] == "string"
    assert properties["days"]["type"] == "integer"
    assert properties["tags"]["type"] == "array"


def test_literal_becomes_an_enum():
    # An enum makes an invalid value impossible rather than discouraged.
    assert sample.parameters["properties"]["units"]["enum"] == ["c", "f"]


def test_only_parameters_without_defaults_are_required():
    assert sample.parameters["required"] == ["city"]


def test_docstring_supplies_descriptions():
    assert sample.description == "Get a forecast."
    assert sample.parameters["properties"]["city"]["description"] == "City name."


def test_schema_matches_the_provider_tool_format():
    schema = sample.schema()
    assert schema["type"] == "function"
    assert set(schema["function"]) == {"name", "description", "parameters"}


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "missing required"),
        ({"city": "X", "bogus": 1}, "unknown argument"),
        ({"city": "X", "days": "three"}, "must be an integer"),
        ({"city": "X", "units": "kelvin"}, "must be one of"),
        ({"city": 5}, "must be a string"),
    ],
)
def test_invalid_arguments_produce_a_recoverable_message(arguments, fragment):
    # The agent must be able to READ the error and correct itself.
    result = ToolRegistry().register(sample).execute(ToolCall("1", "sample", arguments))
    assert not result.ok
    assert fragment in result.error


def test_valid_arguments_run_the_tool():
    result = ToolRegistry().register(sample).execute(ToolCall("1", "sample", {"city": "Berlin"}))
    assert result.ok
    assert "Berlin" in result.content


def test_unknown_tool_lists_the_available_ones():
    result = ToolRegistry().register(calculator).execute(ToolCall("1", "nope", {}))
    assert not result.ok
    assert "calculator" in result.error


def test_duplicate_registration_is_rejected():
    # A silent overwrite means the agent calls a different function than the
    # one you registered.
    registry = ToolRegistry().register(calculator)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(calculator)


def test_tool_exceptions_become_observations_not_crashes():
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    result = ToolRegistry().register(explode).execute(ToolCall("1", "explode", {}))
    assert not result.ok
    assert "boom" in result.error


# ---------------------------------------------------------------------------
# Sandboxes
# ---------------------------------------------------------------------------


def test_calculator_computes_correctly():
    assert safe_eval("1234 * 5678 / 2") == pytest.approx(3503326.0)
    assert safe_eval("-(3 + 4) ** 2") == -49


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",  # module access
        "open('/etc/passwd')",  # file access
        "(1).__class__",  # attribute traversal
        "undefined_name + 1",  # name resolution
        "[x for x in range(3)]",  # comprehension
    ],
)
def test_calculator_rejects_everything_that_is_not_arithmetic(expression):
    # Structural rejection, not a denylist: a denylist of bad strings is
    # always incomplete.
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_calculator_rejects_resource_exhaustion():
    # Not a maths error -- an unbounded exponent is a denial of service.
    with pytest.raises(ValueError, match="exponent too large"):
        safe_eval("2 ** 10 ** 10")


@pytest.mark.parametrize(
    "path", ["../outside.txt", "../../etc/passwd", "/etc/passwd", "sub/../../outside.txt"]
)
def test_file_tool_blocks_escapes_from_its_jail(tmp_path, path):
    (tmp_path / "inside.txt").write_text("safe")
    (tmp_path / "sub").mkdir()
    (tmp_path.parent / "outside.txt").write_text("secret")

    registry = ToolRegistry().register(make_read_file_tool(str(tmp_path)))
    result = registry.execute(ToolCall("1", "read_file", {"path": path}))
    assert not result.ok
    assert "secret" not in (result.content or "")


def test_file_tool_reads_files_inside_the_jail(tmp_path):
    (tmp_path / "inside.txt").write_text("safe content")
    registry = ToolRegistry().register(make_read_file_tool(str(tmp_path)))
    result = registry.execute(ToolCall("1", "read_file", {"path": "inside.txt"}))
    assert result.ok
    assert "safe content" in result.content


def test_file_tool_blocks_symlink_escape(tmp_path):
    # A string check for "../" misses this entirely; resolving the path first
    # is the only correct jail.
    (tmp_path.parent / "outside.txt").write_text("secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(tmp_path.parent / "outside.txt")
    except OSError:  # pragma: no cover - platform without symlink permission
        pytest.skip("symlinks unavailable")

    registry = ToolRegistry().register(make_read_file_tool(str(tmp_path)))
    result = registry.execute(ToolCall("1", "read_file", {"path": "link.txt"}))
    assert not result.ok


@pytest.fixture()
def db():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE orders (id INT, customer TEXT, total REAL)")
    connection.executemany(
        "INSERT INTO orders VALUES (?,?,?)", [(1, "acme", 99.5), (2, "globex", 12.0)]
    )
    return connection


def test_sql_tool_runs_reads(db):
    registry = ToolRegistry().register(make_sql_tool(db))
    result = registry.execute(ToolCall("1", "sql_query", {"query": "SELECT * FROM orders"}))
    assert result.ok
    assert "acme" in result.content


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET total = 0",
        "INSERT INTO orders VALUES (3,'x',1)",
        "SELECT 1; DROP TABLE orders",
    ],
)
def test_sql_tool_rejects_writes_and_leaves_data_intact(db, query):
    registry = ToolRegistry().register(make_sql_tool(db))
    result = registry.execute(ToolCall("1", "sql_query", {"query": query}))
    assert not result.ok
    assert db.execute("SELECT count(*) FROM orders").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# ReAct parsing
# ---------------------------------------------------------------------------


def test_parses_a_well_formed_step():
    parsed = parse_react(
        'Thought: compute\nAction: calculator\nAction Input: {"expression": "2+2"}'
    )
    assert parsed.action == "calculator"
    assert parsed.action_input == {"expression": "2+2"}


def test_final_answer_takes_precedence_over_a_trailing_action():
    # Models often emit a last Thought beside the answer; treating that as
    # another action would loop forever.
    assert parse_react("Thought: x\nAction: foo\nFinal Answer: done").final_answer == "done"


@pytest.mark.parametrize(
    "raw",
    [
        'Action: calc\nAction Input: ```json\n{"a": 1}\n```',
        'Action: calc\nAction Input: {"a": 1,}',
        "Action: calc\nAction Input: {'a': 1}",
    ],
    ids=["fenced", "trailing-comma", "single-quotes"],
)
def test_parser_tolerates_what_models_actually_emit(raw):
    assert parse_react(raw).action_input == {"a": 1}


def test_unparseable_output_is_reported_not_guessed():
    assert parse_react("I'll just answer directly").parse_error


# ---------------------------------------------------------------------------
# The loop: termination guarantees
# ---------------------------------------------------------------------------


REPEAT = 'Thought: thinking\nAction: calculator\nAction Input: {"expression": "1+1"}'


@pytest.fixture()
def registry():
    return ToolRegistry().register(calculator, make_search_tool({"Leave": "25 days"}))


def test_completes_a_multi_step_task(registry):
    script = [
        'Thought: look it up\nAction: search\nAction Input: {"query": "leave"}',
        'Thought: now halve it\nAction: calculator\nAction Input: {"expression": "25/2"}',
        "Thought: done\nFinal Answer: 25 days, half is 12.5",
    ]
    result = ReActAgent(FakeProvider(responses=script), registry).run("leave days?")

    assert result.succeeded
    assert result.stop_reason == "final_answer"
    assert result.tools_used == ["search", "calculator"]


def test_stops_at_the_iteration_cap(registry):
    # Distinct arguments each turn, so loop detection cannot fire first.
    counter = itertools.count()
    provider = FakeProvider(
        respond_fn=lambda m: (
            f'Thought: go\nAction: calculator\nAction Input: {{"expression": "{next(counter)}+1"}}'
        )
    )
    result = ReActAgent(provider, registry, max_iterations=6).run("x")

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 6
    assert not result.succeeded


def test_detects_a_repeating_action(registry):
    result = ReActAgent(FakeProvider(responses=[REPEAT]), registry, max_iterations=20).run("x")
    assert result.stop_reason == "loop_detected"
    assert result.iterations < 20, "must stop well before the cap"


def test_enforces_the_cost_ceiling(registry):
    result = ReActAgent(
        FakeProvider(responses=[REPEAT]), registry, max_iterations=50, max_cost_usd=1e-7
    ).run("x")
    assert result.stop_reason == "budget_exceeded"


def test_enforces_the_wall_clock_deadline(registry):
    result = ReActAgent(
        FakeProvider(responses=[REPEAT], latency=0.02),
        registry,
        max_iterations=50,
        deadline_seconds=0.05,
    ).run("x")
    assert result.stop_reason == "deadline_exceeded"


def test_gives_up_when_output_is_never_parseable(registry):
    result = ReActAgent(FakeProvider(responses=["just chatting"]), registry, max_iterations=20).run(
        "x"
    )
    assert result.stop_reason == "no_progress"


def test_recovers_from_its_own_bad_arguments(registry):
    # The reason ToolResult carries failures as values instead of raising.
    script = [
        'Thought: try\nAction: calculator\nAction Input: {"wrong": "x"}',
        'Thought: fix it\nAction: calculator\nAction Input: {"expression": "6*7"}',
        "Thought: done\nFinal Answer: 42",
    ]
    result = ReActAgent(FakeProvider(responses=script), registry).run("x")
    assert result.succeeded
    assert result.answer == "42"


def test_failed_runs_are_never_reported_as_success(registry):
    result = ReActAgent(FakeProvider(responses=[REPEAT]), registry, max_iterations=20).run("x")
    assert not result.succeeded
    # Best-effort text is still returned, so a UI can show partial progress.
    assert result.answer


def test_trace_records_the_whole_trajectory(registry):
    script = [
        'Thought: search\nAction: search\nAction Input: {"query": "leave"}',
        "Thought: done\nFinal Answer: 25 days",
    ]
    result = ReActAgent(FakeProvider(responses=script), registry).run("x")
    trace = result.trace()
    assert "THOUGHT" in trace and "ACTION" in trace and "OBSERVE" in trace


def test_on_step_callback_streams_progress(registry):
    seen = []
    script = ["Thought: done\nFinal Answer: ok"]
    ReActAgent(FakeProvider(responses=script), registry, on_step=seen.append).run("x")
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_window_memory_never_evicts_the_system_message():
    memory = WindowMemory(k=2)
    memory.add(system("you are an agent"))
    for index in range(10):
        memory.add(user(f"message {index}"))

    messages = memory.messages()
    assert messages[0].role == "system"
    assert len(messages) == 3


def test_token_window_trims_by_size_not_count():
    memory = TokenWindowMemory(max_tokens=100)
    memory.add(system("sys"))
    memory.add(user("B" * 2000))
    memory.add(user("recent"))

    messages = memory.messages()
    assert estimate_tokens(messages) <= 100
    assert any("recent" in m.content for m in messages)


def test_summary_memory_compresses_and_retains_the_summary():
    memory = SummaryMemory(
        summarize_fn=lambda ms: f"[summary of {len(ms)}]", threshold_tokens=50, keep_recent=2
    )
    memory.add(system("sys"))
    for index in range(12):
        memory.add(user(f"turn {index} " + "y" * 60))

    assert memory.summarizations >= 1
    assert any("summary of" in m.content for m in memory.messages())


def test_vector_memory_recalls_by_relevance_not_recency():
    memory = VectorMemory(top_k=2)
    memory.add(user("my favourite colour is blue"))
    for index in range(20):
        memory.add(user(f"unrelated chatter {index}"))

    recalled = memory.recall("what colour do I like")
    assert any("blue" in m.content for m in recalled)


def test_budget_truncates_a_giant_message_rather_than_dropping_everything():
    budget = ConversationBudget(max_tokens=500, max_message_tokens=50)
    messages = [system("sys"), user("A" * 10000), user("important recent question")]

    kept = budget.apply(messages)
    assert estimate_tokens(kept) <= 500
    assert any("important recent" in m.content for m in kept)
    assert any("truncated" in m.content for m in kept)


def test_buffer_memory_keeps_everything():
    memory = BufferMemory()
    for index in range(5):
        memory.add(Message("user", f"m{index}"))
    assert len(memory.messages()) == 5
