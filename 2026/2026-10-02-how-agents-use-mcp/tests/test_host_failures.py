"""What the host does when a server misbehaves: it reports the problem and never hangs."""

from __future__ import annotations

import io
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from conftest import run_async

import agent

FIXTURES = Path(__file__).parent / "fixtures"


def test_a_server_that_fails_to_start_is_reported_by_name_not_as_an_exception_group() -> None:
    servers = {"incidents": agent.SERVERS["incidents"], "broken": FIXTURES / "broken_server.py"}

    with pytest.raises(agent.ServerError, match=r"'broken'.*initialize") as caught:
        run_async(agent.run_agent(agent.SAMPLE_REQUEST, "rules", tracer=agent.Tracer(io.StringIO()), servers=servers))

    assert "broken_server.py" in str(caught.value)  # tells the user what to run to see the real error


def test_a_server_that_dies_during_a_call_gives_an_error_result_and_the_host_keeps_going() -> None:
    async def scenario() -> tuple[agent.ToolCall, agent.ToolCall, agent.ToolCall]:
        host = agent.Host(agent.Tracer(io.StringIO()))
        async with AsyncExitStack() as stack:
            await host.connect(stack, servers={"probe": FIXTURES / "probe_server.py", "docs": agent.SERVERS["docs"]})
            await host.discover()
            died = await host.call_tool("crash", {})
            dead_again = await host.call_tool("ping", {})  # same dead server
            other = await host.call_tool("search_docs", {"query": "orders db failover"})  # a healthy server
            return died, dead_again, other

    died, dead_again, other = run_async(scenario())
    assert died.is_error and "Connection closed" in died.text
    assert dead_again.is_error  # fails fast instead of hanging
    assert not other.is_error and other.result[0]["title"] == "Orders database: replica failover"


def test_a_tool_name_offered_by_two_servers_is_reported_at_discovery() -> None:
    servers = {"one": FIXTURES / "probe_server.py", "two": FIXTURES / "probe_server.py"}
    with pytest.raises(agent.ServerError, match="offered by both 'one' and 'two'"):
        run_async(agent.run_agent("show me the runbook", "rules", tracer=agent.Tracer(io.StringIO()), servers=servers))
