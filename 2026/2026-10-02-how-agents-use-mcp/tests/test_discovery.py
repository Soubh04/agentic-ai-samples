"""`tools/list` against each real server: names, descriptions and input schemas."""

from __future__ import annotations

import pytest
from conftest import open_server, run_async

EXPECTED = {
    "incidents": ("search_incidents", {"priority": "string", "days": "integer"}),
    "email": ("send_email", {"to": "string", "subject": "string", "body": "string"}),
    "docs": ("search_docs", {"query": "string"}),
}


@pytest.mark.parametrize("server", sorted(EXPECTED))
def test_tools_list_returns_the_tool_with_its_input_schema(server: str) -> None:
    async def list_tools():
        async with open_server(server) as session:
            return await session.list_tools()

    result = run_async(list_tools())
    tool_name, arguments = EXPECTED[server]

    (tool,) = result.tools  # each server offers exactly one tool
    assert tool.name == tool_name
    assert tool.description and len(tool.description) > 20  # the LLM reads this to decide when to call it
    assert tool.input_schema["type"] == "object"
    assert {name: spec["type"] for name, spec in tool.input_schema["properties"].items()} == arguments
    assert set(tool.input_schema["required"]) == set(arguments)
    assert all(spec.get("description") for spec in tool.input_schema["properties"].values())
