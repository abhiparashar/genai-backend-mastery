from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.deps import get_budgets, reset_provider, set_provider  # noqa: E402
from app.llm import FakeProvider  # noqa: E402
from app.observability import METRICS  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_state():
    """Reset every piece of process-wide state between tests.

    The service holds singletons on purpose (provider, budgets, metrics,
    cached settings). Without this fixture tests leak into each other and
    fail depending on ORDER, which is the worst kind of flaky suite.
    """
    get_settings.cache_clear()
    get_budgets().reset()
    METRICS.reset()
    reset_provider()
    yield
    get_settings.cache_clear()
    get_budgets().reset()
    METRICS.reset()
    reset_provider()


@pytest.fixture()
def client():
    from app.main import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app())


@pytest.fixture()
def auth():
    return {"X-API-Key": "dev-key-1"}


@pytest.fixture()
def body():
    return {"messages": [{"role": "user", "content": "what is our refund policy?"}]}


@pytest.fixture()
def scripted():
    """Install a provider with known output so assertions can be exact."""

    def install(**kwargs):
        provider = FakeProvider(**kwargs)
        set_provider(provider)
        return provider

    return install
