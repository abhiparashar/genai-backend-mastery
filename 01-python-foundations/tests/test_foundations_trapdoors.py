"""Every trapdoor, tested against BOTH the reference and your attempt.

Each behaviour gets one test function parametrized over two modules:

    solutions/  must pass  -- proves the reference implementation is correct
    exercises/  skips      -- until you implement it, then it must pass

So `pytest 01-python-foundations/tests -q` is a progress bar, not a wall of
red. Run it as you work.
"""

from __future__ import annotations

import pytest
from conftest import attempt
from exercises import ex01_java_trapdoors as ex
from solutions import sol01_java_trapdoors as sol

# Both modules expose the same API, so one test covers both.
MODULES = [pytest.param(sol, id="solution"), pytest.param(ex, id="your-attempt")]


@pytest.fixture(params=MODULES)
def mod(request):
    return request.param


# ---------------------------------------------------------------------------
# 1. Mutable default arguments
# ---------------------------------------------------------------------------


def test_default_argument_is_not_shared_between_calls(mod):
    # The whole bug: a shared default accumulates across calls.
    first = attempt(mod.add_item, "a")
    second = attempt(mod.add_item, "b")

    assert first == ["a"]
    assert second == ["b"], "a mutable default leaks state between calls"


def test_supplied_basket_is_still_used(mod):
    assert attempt(mod.add_item, "c", ["existing"]) == ["existing", "c"]


# ---------------------------------------------------------------------------
# 2. is vs ==
# ---------------------------------------------------------------------------


def test_equal_but_distinct_objects(mod):
    equal, identical = attempt(mod.compare_values, [1, 2], [1, 2])
    assert equal is True
    assert identical is False, "two distinct lists are equal but not the same object"


def test_same_object_is_both(mod):
    value = [1, 2]
    assert attempt(mod.compare_values, value, value) == (True, True)


# ---------------------------------------------------------------------------
# 3. Copying
# ---------------------------------------------------------------------------


def test_shallow_copy_shares_nested_objects(mod):
    original = [[1, 2], [3, 4]]
    shallow = attempt(mod.copy_matrix, original, False)
    shallow[0][0] = 99
    # The outer list is new; the inner lists are the SAME objects.
    assert original[0][0] == 99


def test_deep_copy_is_independent(mod):
    original = [[1, 2], [3, 4]]
    deep = attempt(mod.copy_matrix, original, True)
    deep[0][0] = 99
    assert original[0][0] == 1


# ---------------------------------------------------------------------------
# 4. Closures
# ---------------------------------------------------------------------------


def test_each_closure_captures_its_own_value(mod):
    multipliers = attempt(mod.make_multipliers)
    assert [f(10) for f in multipliers] == [0, 10, 20, 30, 40]


def test_the_broken_version_demonstrates_late_binding():
    # Only the solution module ships the deliberately-broken example.
    assert sol.make_multipliers_broken()[0](10) == 40, "all five capture the final i"


# ---------------------------------------------------------------------------
# 5. Truthiness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(0, 0), (None, 30), (5, 5), (120, 120)],
    ids=["explicit-zero", "unset", "small", "large"],
)
def test_zero_is_a_real_value_not_a_missing_one(mod, configured, expected):
    # `configured or 30` passes every case here EXCEPT the first, which is
    # exactly why this bug survives review.
    assert attempt(mod.get_timeout, configured) == expected


# ---------------------------------------------------------------------------
# 6. equals / hashCode
# ---------------------------------------------------------------------------


def test_point_compares_by_value(mod):
    assert mod.Point(1, 2) == mod.Point(1, 2)
    assert mod.Point(1, 2) != mod.Point(2, 1)


def test_point_is_usable_as_a_set_member(mod):
    # Defining __eq__ without __hash__ makes this raise TypeError.
    try:
        deduplicated = {mod.Point(1, 2), mod.Point(1, 2), mod.Point(3, 4)}
    except TypeError:
        pytest.skip("not attempted yet: Point is unhashable")
    assert len(deduplicated) == 2


def test_the_broken_class_shows_why_the_contract_matters():
    with pytest.raises(TypeError):
        {sol.BadPoint(1, 2)}  # noqa: B018 - raising IS the assertion


# ---------------------------------------------------------------------------
# 7. EAFP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8080", 8080),
        ("1", 1),
        ("65535", 65535),
        ("0", None),
        ("70000", None),
        ("-5", None),
        ("nope", None),
        ("", None),
        ("80.5", None),
    ],
)
def test_port_parsing_handles_every_bad_input(mod, raw, expected):
    assert attempt(mod.parse_port, raw) == expected


# ---------------------------------------------------------------------------
# 8. Iteration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "target", "expected"),
    [
        (["a", "b", "a"], "a", [0, 2]),
        ([], "a", []),
        (["x"], "y", []),
        ([1, 1, 1], 1, [0, 1, 2]),
    ],
)
def test_finds_every_index(mod, items, target, expected):
    assert attempt(mod.index_of_all, items, target) == expected


def test_zip_stops_at_the_shorter_sequence(mod):
    # Silently. If that would hide a bug, assert the lengths yourself.
    assert attempt(mod.pair_up, ["a", "b"], [1, 2, 3]) == [("a", 1), ("b", 2)]


# ---------------------------------------------------------------------------
# 9. Comprehensions
# ---------------------------------------------------------------------------


def test_filters_maps_and_sorts_in_one_pass(mod):
    users = [
        {"email": "b@x.com", "active": True},
        {"email": "a@x.com", "active": True},
        {"email": "c@x.com", "active": False},
    ]
    assert attempt(mod.active_user_emails, users) == ["a@x.com", "b@x.com"]


def test_no_active_users_yields_an_empty_list(mod):
    assert attempt(mod.active_user_emails, [{"email": "a@x.com", "active": False}]) == []


# ---------------------------------------------------------------------------
# 10. Higher-order functions
# ---------------------------------------------------------------------------


def test_functions_are_applied_in_order(mod):
    # Order matters: (3+1)*2 = 8, not 3+(1*2) = 5.
    assert attempt(mod.apply_all, 3, [lambda x: x + 1, lambda x: x * 2]) == 8


def test_an_empty_pipeline_returns_the_input(mod):
    assert attempt(mod.apply_all, 5, []) == 5
