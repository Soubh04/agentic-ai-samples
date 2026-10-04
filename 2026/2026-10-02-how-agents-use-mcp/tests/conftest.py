"""Shared helpers for the tests.

Everything here talks to the REAL server scripts over REAL stdio. Nothing is mocked on
the MCP side. The only fake in the suite is the Anthropic HTTP layer (see
test_claude_planner.py), so the tests never need a network connection or an API key.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client

import agent

T = TypeVar("T")
TEST_TIMEOUT_SECONDS = 90


def run_async(awaitable: Awaitable[T]) -> T:
    """Run a coroutine to completion; fail instead of hanging if a server stalls."""

    async def bounded() -> T:
        return await asyncio.wait_for(awaitable, timeout=TEST_TIMEOUT_SECONDS)

    return asyncio.run(bounded())


@asynccontextmanager
async def open_server(name: str) -> AsyncIterator[ClientSession]:
    """Start ONE server script as a subprocess, run `initialize`, and yield the session."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(agent.SERVERS[name])],
        env={key: os.environ[key] for key in agent.FORWARDED_ENV if key in os.environ},
    )
    async with (
        stdio_client(params, errlog=sys.__stderr__ or sys.stderr) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream, read_timeout_seconds=30) as session,
    ):
        await session.initialize()
        yield session


def read_outbox(path: Path) -> list[dict[str, Any]]:
    """The messages the email server wrote, one JSON object per line."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


@pytest.fixture
def outbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throw-away outbox path, handed to the email server through EMAIL_OUTBOX_PATH."""
    path = tmp_path / "mail" / "outbox.jsonl"  # the directory does not exist yet on purpose
    monkeypatch.setenv("EMAIL_OUTBOX_PATH", str(path))
    return path
