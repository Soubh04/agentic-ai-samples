"""Test setup. No test may reach a real API: keys are removed and network connections are blocked."""
import socket

import pytest

ENV_VARS = (
    "TYPESAFE_API_KEY",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_DEFAULT_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_PROFILE",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Every test starts with no API keys, even if the developer has some exported."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture
def fake_keys(monkeypatch):
    """Placeholder values so the SDK clients can be constructed. They are not credentials."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder-typesafe")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-anthropic")
