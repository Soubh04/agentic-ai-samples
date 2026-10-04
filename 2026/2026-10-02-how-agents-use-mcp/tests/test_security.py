"""What the host hands to the servers it starts."""

from __future__ import annotations

import io
import json
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from conftest import run_async

import agent

PROBE = Path(__file__).parent / "fixtures" / "probe_server.py"


def test_servers_do_not_inherit_the_hosts_environment_or_its_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("SOME_OTHER_SECRET", "also-not-real")
    monkeypatch.setenv("EMAIL_OUTBOX_PATH", str(tmp_path / "outbox.jsonl"))

    async def visible_to_server() -> list[str]:
        host = agent.Host(agent.Tracer(io.StringIO()))
        async with AsyncExitStack() as stack:
            await host.connect(stack, servers={"probe": PROBE})
            call = await host.sessions["probe"].call_tool("env_names", {})
            return json.loads(call.content[0].text)

    names = run_async(visible_to_server())

    assert "EMAIL_OUTBOX_PATH" in names  # explicitly forwarded
    assert "PATH" in names  # part of the SDK's small allow-list
    assert "ANTHROPIC_API_KEY" not in names
    assert "SOME_OTHER_SECRET" not in names
