from __future__ import annotations

import pathlib
import sys

import pytest

# `01-python-foundations` starts with a digit, so it is not importable as a
# package. Put the module root on sys.path instead.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundations_harness import attempt  # noqa: E402


@pytest.fixture()
def check():
    """Fixture form of `attempt`, for readability inside tests."""
    return attempt
