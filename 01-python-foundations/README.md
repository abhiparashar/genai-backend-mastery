# Module 01 — Python Foundations

**Phase 1 of the roadmap · Weeks 1–6**

Not "Python basics". This module targets the specific places where a **correct Java instinct produces wrong Python** — the bugs you will write in your first month, before anyone warns you.

```bash
pytest 01-python-foundations/tests -q
```

Unattempted exercises show as **skipped**, finished ones as **passed**. That's your progress bar, not a wall of red.

```
.s.s.s.s.s.s.s.....ssss...s..........sssssssss....ssss.s.s.s.s
 ↑ solution passes    ↑ your attempt, not yet written
```

Each behaviour is tested twice — once against `solutions/` (proving the reference is correct) and once against `exercises/` (skipping until you fill it in). Solve one function and its tests turn green immediately.

**Attempt first, read the solution second.** The solutions file is a good read, but reading it first converts an exercise into a tutorial.

---

## The ten trapdoors

| # | Trap | Why Java didn't prepare you |
|---|---|---|
| 1 | **Mutable default arguments** | Java re-evaluates defaults per call; Python evaluates **once at definition**, so callers share one list |
| 2 | **`is` vs `==`** | Inverted from Java: there `==` on Strings is the bug, here `is` is — and it *appears* to work because small ints and short strings are interned |
| 3 | **Shallow vs deep copy** | `list(x)`, `x[:]`, `x.copy()` are all shallow, like `clone()` on an array of references |
| 4 | **Late-binding closures** | Java requires captured locals to be effectively final, making this a compile error there and a silent bug here |
| 5 | **Truthiness** | `0`, `""`, `[]`, `None` are all falsy, so `value or default` silently eats a legitimate `0` |
| 6 | **`__eq__` / `__hash__`** | Python *enforces* at runtime the contract Java only documents: define `__eq__` alone and instances become unhashable |
| 7 | **EAFP vs LBYL** | Checked exceptions train "validate first"; Python prefers "try it, handle failure" — faster and race-free |
| 8 | **`enumerate` / `zip`** | `for i in range(len(x))` is a Java habit; also, `zip` silently stops at the shorter input |
| 9 | **Class-level mutables** | The equivalent of `static`, but expressed as mere indentation |
| 10 | **Comprehensions** | Replace the whole `stream().filter().map().sorted()` pipeline, and run faster than a hand-written loop |

---

## Java → Python cheat sheet

| Java | Python |
|---|---|
| `ArrayList<T>` | `list` |
| `HashMap<K,V>` | `dict` |
| `HashSet<T>` | `set` |
| `Optional<T>` | `Optional[T]` / `None` |
| `record` | `@dataclass(frozen=True)` |
| `interface` | `Protocol` (structural) or `ABC` (nominal) |
| `stream().filter().map()` | `[f(x) for x in xs if cond]` |
| `Collectors.groupingBy` | `defaultdict(list)` / `setdefault` |
| `Optional.orElse` | `value if value is not None else default` |
| `equals` / `hashCode` | `__eq__` / `__hash__` |
| `toString` | `__repr__` (and `__str__`) |
| `static` field | class attribute |
| `final` | naming convention (`UPPER_CASE`) — not enforced |
| checked exceptions | **don't exist** — nothing forces you to handle anything |
| `try-with-resources` | `with` (context manager) |
| `null` | `None` |
| `System.out.println` | `print` |
| `StringBuilder` | `"".join(parts)` |

### The three that bite hardest

**No checked exceptions.** Nothing tells you a function can raise. Read the docs, and use `try/except` around genuine boundaries rather than everywhere.

**No `final`, no `private`.** `_name` is a convention meaning "internal"; nothing enforces it. Discipline replaces the compiler.

**Indentation is syntax.** Mixing tabs and spaces is an error, not a style debate. Let the formatter handle it — `ruff format`.

---

## The mistakes to unlearn

```python
# Writing Java in Python
class UserService:
    def __init__(self):
        self._users = []

    def get_users(self):  # a getter -- unnecessary
        return self._users

    def set_users(self, users):  # a setter -- unnecessary
        self._users = users


# Python: just use the attribute. If you later need logic, @property
# adds it WITHOUT changing a single caller. That's why getters are
# pointless here and load-bearing in Java.
class UserService:
    def __init__(self):
        self.users = []
```

| Instead of | Write |
|---|---|
| `for i in range(len(items))` | `for item in items` / `enumerate` |
| `if len(items) > 0:` | `if items:` |
| `if x == None:` | `if x is None:` |
| `result = ""` then `+=` in a loop | `"".join(parts)` |
| a getter/setter pair | a plain attribute, `@property` later |
| `type(x) == Foo` | `isinstance(x, Foo)` |
| `dict.keys()` just to iterate | `for key in dict` |
| `try: ... except: pass` | catch a **specific** exception |

---

## Files

| Path | Purpose |
|---|---|
| `exercises/ex01_java_trapdoors.py` | Stubs to fill in. Each raises `NotImplementedError` with a hint. |
| `solutions/sol01_java_trapdoors.py` | Reference implementations, each with the *why*. Includes deliberately broken versions (`add_item_broken`, `make_multipliers_broken`, `BadPoint`) so you can see the bug behave. |
| `tests/test_foundations_trapdoors.py` | One suite, run against both modules. |
| `tests/conftest.py` | The `attempt()` helper that turns `NotImplementedError` into a skip. |

> One detail worth noticing in `solutions/`: writing the `is`-vs-`==` doctest as `compare_values(1000, 1000)` **failed** — CPython constant-folds identical literals in one code object, so `is` returned `True`. That is exactly why "never use `is` for value comparison" is an absolute rule rather than a judgement call, and the comment in the file records it.

---

## Exit criteria

1. All exercise tests pass.
2. You write comprehensions without translating from a loop first.
3. You reach for `dict.get`, `defaultdict` and `Counter` reflexively.
4. You stop writing getters and setters.
5. You can explain why `def f(x=[])` is a bug **and** why it's not in Java.

**Next:** [Module 02 — Advanced Python](../02-python-advanced/) — decorators, generators, and the asyncio that every LLM call depends on.
