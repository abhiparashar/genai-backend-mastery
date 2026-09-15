"""EXERCISES: the ten Python behaviours that catch every Java developer.

Fill in each function. Run the tests to see your progress:

    pytest 01-python-foundations/tests -q

Unattempted exercises show as SKIPPED, finished ones as PASSED. That is your
progress bar. The reference implementations are in
`solutions/sol01_java_trapdoors.py` -- read them AFTER attempting, not before.

Every function here corresponds to a real bug a Java developer writes at
least once in their first month of Python.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

TODO = "replace this line with your implementation"


# ---------------------------------------------------------------------------
# 1. Mutable default arguments
# ---------------------------------------------------------------------------


def add_item(item: str, basket: Optional[list] = None) -> list:
    """Append `item` to `basket` and return it. A fresh list when none given.

    The trap: `def f(basket=[])` evaluates the default ONCE at definition
    time, so every caller shares one list. Java re-evaluates defaults per
    call, so this instinct never develops.

    >>> add_item("a")
    ['a']
    >>> add_item("b")
    ['b']
    >>> add_item("c", ["existing"])
    ['existing', 'c']
    """
    raise NotImplementedError("ex01.1: use a None sentinel, then build inside")


# ---------------------------------------------------------------------------
# 2. `is` vs `==`
# ---------------------------------------------------------------------------


def compare_values(a: Any, b: Any) -> tuple[bool, bool]:
    """Return `(values are equal, same object)`.

    >>> compare_values([1, 2], [1, 2])
    (True, False)
    >>> value = [1, 2]
    >>> compare_values(value, value)
    (True, True)
    """
    raise NotImplementedError("ex01.2: one comparison uses ==, the other uses is")


# ---------------------------------------------------------------------------
# 3. Shallow vs deep copy
# ---------------------------------------------------------------------------


def copy_matrix(matrix: list, deep: bool) -> list:
    """Copy a nested list. `deep=False` shares the inner lists.

    >>> original = [[1, 2], [3, 4]]
    >>> shallow = copy_matrix(original, deep=False)
    >>> shallow[0][0] = 99
    >>> original[0][0]
    99
    >>> original = [[1, 2], [3, 4]]
    >>> deep = copy_matrix(original, deep=True)
    >>> deep[0][0] = 99
    >>> original[0][0]
    1
    """
    raise NotImplementedError("ex01.3: list() is shallow; the copy module has the other")


# ---------------------------------------------------------------------------
# 4. Late-binding closures
# ---------------------------------------------------------------------------


def make_multipliers() -> list:
    """Return 5 functions where function `i` multiplies its argument by `i`.

    The trap: `[lambda x: x * i for i in range(5)]` returns five functions
    that ALL multiply by 4, because the closure captures the variable, not
    its value.

    >>> [f(10) for f in make_multipliers()]
    [0, 10, 20, 30, 40]
    """
    raise NotImplementedError("ex01.4: bind the current value as a default argument")


# ---------------------------------------------------------------------------
# 5. Truthiness
# ---------------------------------------------------------------------------


def get_timeout(configured: Optional[int]) -> int:
    """Return `configured`, or 30 when it is None. An explicit 0 stays 0.

    The trap: `configured or 30` turns a deliberate 0 into 30, because 0 is
    falsy. This bug reaches production constantly.

    >>> get_timeout(0)
    0
    >>> get_timeout(None)
    30
    >>> get_timeout(5)
    5
    """
    raise NotImplementedError("ex01.5: test for None explicitly, not truthiness")


# ---------------------------------------------------------------------------
# 6. equals / hashCode
# ---------------------------------------------------------------------------


@dataclass
class Point:
    """Make `Point` compare by value AND work as a set member / dict key.

    Java's equals/hashCode contract, enforced at runtime: define `__eq__`
    alone and Python sets `__hash__ = None`, making instances unhashable.

    Hint: one decorator argument does both.

    >>> Point(1, 2) == Point(1, 2)
    True
    >>> len({Point(1, 2), Point(1, 2)})
    1
    """

    x: int
    y: int


# ---------------------------------------------------------------------------
# 7. EAFP vs LBYL
# ---------------------------------------------------------------------------


def parse_port(raw: str) -> Optional[int]:
    """Parse a TCP port (1-65535). Return None if invalid.

    Write it EAFP: try the conversion and handle the failure, rather than
    pre-validating with `.isdigit()`. Java's checked exceptions train the
    opposite instinct.

    >>> parse_port("8080")
    8080
    >>> parse_port("nope") is None
    True
    >>> parse_port("0") is None
    True
    >>> parse_port("70000") is None
    True
    """
    raise NotImplementedError("ex01.7: try/except ValueError, then range-check")


# ---------------------------------------------------------------------------
# 8. Iteration
# ---------------------------------------------------------------------------


def index_of_all(items: Sequence[Any], target: Any) -> list:
    """Indices of every occurrence of `target`.

    Use `enumerate`, not `range(len(items))`.

    >>> index_of_all(["a", "b", "a"], "a")
    [0, 2]
    >>> index_of_all([], "a")
    []
    """
    raise NotImplementedError("ex01.8: enumerate gives you index and value together")


def pair_up(names: Sequence[str], scores: Sequence[int]) -> list:
    """Pair the sequences elementwise, stopping at the shorter one.

    >>> pair_up(["a", "b"], [1, 2, 3])
    [('a', 1), ('b', 2)]
    """
    raise NotImplementedError("ex01.9: one builtin does this")


# ---------------------------------------------------------------------------
# 9. Comprehensions
# ---------------------------------------------------------------------------


def active_user_emails(users: Sequence[dict]) -> list:
    """Emails of active users, sorted. The Java Streams pipeline in one line.

    >>> active_user_emails([
    ...     {"email": "b@x.com", "active": True},
    ...     {"email": "a@x.com", "active": True},
    ...     {"email": "c@x.com", "active": False},
    ... ])
    ['a@x.com', 'b@x.com']
    """
    raise NotImplementedError("ex01.10: a comprehension with an if, wrapped in sorted()")


# ---------------------------------------------------------------------------
# 10. Higher-order functions
# ---------------------------------------------------------------------------


def apply_all(value: int, functions: Sequence[Callable[[int], int]]) -> int:
    """Apply each function to `value` in order, returning the final result.

    >>> apply_all(3, [lambda x: x + 1, lambda x: x * 2])
    8
    >>> apply_all(5, [])
    5
    """
    raise NotImplementedError("ex01.11: fold the value through the list")
