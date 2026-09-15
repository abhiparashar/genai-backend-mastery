from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable

import pytest

# `01-python-foundations` starts with a digit, so it is not importable as a
# package. Put the module root on sys.path instead.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def attempt(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a learner's function, skipping the test if it is unattempted.

    This is what turns the suite into a progress bar. Every test runs twice:
    once against the reference solution (must pass, so the suite proves the
    reference is correct) and once against the learner's exercise, which
    SKIPS until they fill it in.

    Without this, a learner starting the module sees 40 red failures, which
    is demoralising and tells them nothing. With it, they see 40 skips that
    turn green one at a time.
    """
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"not attempted yet: {exc}")


@pytest.fixture()
def check():
    """Fixture form of `attempt`, for readability inside tests."""
    return attempt
