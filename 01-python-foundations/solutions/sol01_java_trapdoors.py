"""The ten Python behaviours that catch every Java developer.

Not "Python basics" -- the specific places where a correct Java instinct
produces wrong Python. Each function below is a bug a Java developer will
write at least once, plus the idiom that avoids it.

Work `exercises/ex01_java_trapdoors.py`, then read this.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# 1. Mutable default arguments
# ---------------------------------------------------------------------------


def add_item_broken(item: str, basket: list = []) -> list:  # noqa: B006
    """THE classic. Deliberately wrong -- see `add_item` for the fix.

    The default is evaluated ONCE, when the function is defined, not on each
    call. So every caller that omits `basket` shares the same list forever.

    >>> add_item_broken("a")
    ['a']
    >>> add_item_broken("b")
    ['a', 'b']
    """
    basket.append(item)
    return basket


def add_item(item: str, basket: Optional[list] = None) -> list:
    """The fix: `None` sentinel, build inside.

    In Java a default parameter value is re-evaluated per call, so this trap
    has no equivalent and the instinct never develops.

    >>> add_item("a")
    ['a']
    >>> add_item("b")
    ['b']
    """
    if basket is None:
        basket = []
    basket.append(item)
    return basket


# ---------------------------------------------------------------------------
# 2. `is` vs `==`
# ---------------------------------------------------------------------------


def compare_values(a: Any, b: Any) -> tuple[bool, bool]:
    """Return `(a == b, a is b)`.

    `==` compares VALUE (Java's `.equals`). `is` compares IDENTITY (Java's
    `==` on references).

    The trap runs opposite to Java: `==` on Strings in Java is the buggy one,
    so developers learn "use .equals". In Python `==` is correct and `is` is
    the bug -- and it *appears* to work, because CPython interns small ints
    (-5..256) and short strings.

    Worse, the interning rules are an IMPLEMENTATION DETAIL you must never
    rely on. Writing this doctest as `compare_values(1000, 1000)` actually
    FAILED here: CPython constant-folds two identical literals in the same
    code object into one object, so `is` returned True. That is precisely why
    the rule is absolute rather than a judgement call.

    Use `is` only for `None`, `True`, `False`.

    >>> compare_values([1, 2], [1, 2])      # distinct objects, equal values
    (True, False)
    >>> value = [1, 2]
    >>> compare_values(value, value)        # same object
    (True, True)
    """
    return a == b, a is b


# ---------------------------------------------------------------------------
# 3. Shallow vs deep copy
# ---------------------------------------------------------------------------


def copy_matrix(matrix: list, deep: bool) -> list:
    """Copy a nested list, shallowly or deeply.

    `list(x)`, `x[:]` and `x.copy()` are all SHALLOW: the outer list is new,
    the inner objects are shared. Mutating `copy[0][0]` also changes the
    original. Same semantics as Java's `clone()` on an array of references,
    and the same surprise.

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
    return copy.deepcopy(matrix) if deep else list(matrix)


# ---------------------------------------------------------------------------
# 4. Late-binding closures
# ---------------------------------------------------------------------------


def make_multipliers_broken() -> list:
    """Deliberately wrong: every function returns x * 4.

    The closure captures the VARIABLE `i`, not its value. By the time any of
    them runs, the loop has finished and `i` is 4.

    Java sidesteps this by requiring captured locals to be effectively final,
    so the mistake is a compile error there and a silent bug here.

    >>> [f(10) for f in make_multipliers_broken()]
    [40, 40, 40, 40, 40]
    """
    # noqa is deliberate: ruff's B023 detects exactly the bug being
    # demonstrated here. That it fires is the lesson -- turn this rule on in
    # your own projects and the linter catches late binding before review does.
    return [lambda x: x * i for i in range(5)]  # noqa: B023


def make_multipliers() -> list:
    """The fix: bind the current value as a default argument.

    >>> [f(10) for f in make_multipliers()]
    [0, 10, 20, 30, 40]
    """
    return [lambda x, factor=i: x * factor for i in range(5)]


# ---------------------------------------------------------------------------
# 5. Truthiness
# ---------------------------------------------------------------------------


def describe_truthiness(value: Any) -> str:
    """Classify a value the way `if value:` would.

    Python has no `boolean` requirement: empty containers, 0, "" and None are
    all falsy. That makes `if not items:` idiomatic -- and makes
    `if not count:` a bug when `count` is legitimately 0.

    When 0 or "" are valid values, test `is None` explicitly.

    >>> describe_truthiness(0)
    'falsy'
    >>> describe_truthiness([])
    'falsy'
    >>> describe_truthiness('0')
    'truthy'
    """
    return "truthy" if value else "falsy"


def get_timeout(configured: Optional[int]) -> int:
    """Return the configured timeout, defaulting to 30 -- 0 stays 0.

    `configured or 30` would be wrong: 0 is falsy, so an explicit "no
    timeout" silently becomes 30.

    >>> get_timeout(0)
    0
    >>> get_timeout(None)
    30
    """
    return 30 if configured is None else configured


# ---------------------------------------------------------------------------
# 6. equals / hashCode -> __eq__ / __hash__
# ---------------------------------------------------------------------------


class BadPoint:
    """Defines `__eq__` without `__hash__`. Deliberately broken.

    Defining `__eq__` sets `__hash__ = None`, making instances unhashable, so
    they cannot go in a set or be dict keys. Python enforces at runtime the
    contract Java only documents: equal objects must hash equally.
    """

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BadPoint) and (self.x, self.y) == (other.x, other.y)


@dataclass(frozen=True)
class Point:
    """The right answer: a frozen dataclass.

    `frozen=True` generates `__eq__` AND `__hash__` consistently, so the
    contract cannot drift. This is Java's `record`, and it replaces about 40
    lines of boilerplate.

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


def parse_port_lbyl(raw: str) -> Optional[int]:
    """Look Before You Leap. Works, but duplicates the parser's own rules.

    >>> parse_port_lbyl("8080")
    8080
    >>> parse_port_lbyl("-5") is None
    True
    """
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if 0 < value <= 65535 else None


def parse_port(raw: str) -> Optional[int]:
    """EAFP -- Easier to Ask Forgiveness than Permission. The Python idiom.

    Try it, handle the failure. It is faster on the happy path (no duplicated
    validation) and race-free, which matters for files and network resources
    where "check then act" can be invalidated in between.

    Java's checked exceptions train the opposite instinct.

    >>> parse_port("8080")
    8080
    >>> parse_port("nope") is None
    True
    """
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if 0 < value <= 65535 else None


# ---------------------------------------------------------------------------
# 8. Iteration
# ---------------------------------------------------------------------------


def index_of_all(items: Sequence[Any], target: Any) -> list:
    """Indices of every occurrence.

    A Java dev writes `for i in range(len(items))`. Python's `enumerate` is
    the idiom: it gives index and value together, with no indexing at all.

    >>> index_of_all(["a", "b", "a"], "a")
    [0, 2]
    """
    return [index for index, value in enumerate(items) if value == target]


def pair_up(names: Sequence[str], scores: Sequence[int]) -> list:
    """Combine two sequences elementwise.

    `zip` stops at the SHORTER input -- silently. If that would hide a bug,
    pass `strict=True` (3.10+) or assert the lengths.

    >>> pair_up(["a", "b"], [1, 2, 3])
    [('a', 1), ('b', 2)]
    """
    return list(zip(names, scores))


# ---------------------------------------------------------------------------
# 9. Mutable class attributes
# ---------------------------------------------------------------------------


class SharedRegistry:
    """Deliberately broken: `items` is shared by every instance.

    A class-level mutable is the equivalent of a Java `static` field. Java
    developers rarely make this mistake in Java -- because `static` is
    explicit there, and here it is just indentation.
    """

    items: list = []

    def add(self, value: str) -> None:
        self.items.append(value)


@dataclass
class Registry:
    """The fix: `field(default_factory=list)` gives each instance its own.

    >>> a, b = Registry(), Registry()
    >>> a.add("x")
    >>> b.items
    []
    """

    items: list = field(default_factory=list)

    def add(self, value: str) -> None:
        self.items.append(value)


# ---------------------------------------------------------------------------
# 10. Comprehensions over Streams
# ---------------------------------------------------------------------------


def active_user_emails(users: Sequence[dict]) -> list:
    """Filter, map, and sort -- the Java Streams pipeline, in one line.

    Java:  users.stream().filter(...).map(...).sorted().toList()
    Python: [expr for u in users if cond]  then sorted()

    Comprehensions are faster than the equivalent loop (the iteration runs in
    C) and are the idiom. Reach for `filter`/`map` only when you already have
    the function to hand.

    >>> active_user_emails([
    ...     {"email": "b@x.com", "active": True},
    ...     {"email": "a@x.com", "active": True},
    ...     {"email": "c@x.com", "active": False},
    ... ])
    ['a@x.com', 'b@x.com']
    """
    return sorted(user["email"] for user in users if user["active"])


def group_by_domain(emails: Sequence[str]) -> dict:
    """Group emails by domain. Java's `Collectors.groupingBy`.

    `setdefault` avoids a KeyError check; `collections.defaultdict(list)` is
    the other idiomatic answer and is better when grouping in a loop.

    >>> group_by_domain(["a@x.com", "b@y.com", "c@x.com"])
    {'x.com': ['a@x.com'], 'y.com': ['b@y.com']}
    """
    groups: dict = {}
    for email in emails:
        domain = email.split("@")[1]
        groups.setdefault(domain, []).append(email)
    # Keep only the first per domain to keep the doctest small and stable.
    return {domain: members[:1] for domain, members in groups.items()}


def apply_all(value: int, functions: Sequence[Callable[[int], int]]) -> int:
    """Apply each function in turn. Java's `Function.andThen` chain.

    >>> apply_all(3, [lambda x: x + 1, lambda x: x * 2])
    8
    """
    for function in functions:
        value = function(value)
    return value
