"""Test harness that turns an unattempted exercise into a skip.

WHY THIS IS NOT IN conftest.py

It used to be, and it broke the whole repo's test collection. Several
modules each have a `tests/conftest.py`, and without `__init__.py` files
pytest imports them all under the bare module name `conftest`. The first one
imported wins in `sys.modules`, so `from conftest import attempt` inside
module 01 resolved to module 06's conftest and raised ImportError.

The fix is a uniquely-named module. Fixtures still belong in conftest.py;
importable HELPERS need a name that cannot collide.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest


def attempt(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a learner's function, skipping the test if it is unattempted.

    This is what makes the suite a progress bar. Every behaviour is tested
    twice: against the reference solution (must pass, proving the reference
    correct) and against the learner's exercise, which SKIPS until filled in.

    Without it, a learner opening the module sees 40 red failures, which is
    demoralising and says nothing. With it, they see skips turning green one
    at a time.
    """
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"not attempted yet: {exc}")
