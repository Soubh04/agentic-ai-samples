"""Small pure functions of the host: result decoding, signatures, the tool catalog."""

from __future__ import annotations

import pytest
from mcp import types

import agent


def text_result(*texts: str, is_error: bool = False, structured=None) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=t) for t in texts],
        is_error=is_error,
        structured_content=structured,
    )


class TestDecodeResult:
    def test_json_text_is_parsed(self) -> None:
        assert agent.decode_result(text_result('[{"id": "INC-1"}]')) == [{"id": "INC-1"}]

    def test_an_empty_json_array_is_an_empty_list(self) -> None:
        assert agent.decode_result(text_result("[]")) == []

    def test_plain_text_stays_text(self) -> None:
        assert agent.decode_result(text_result("not json at all")) == "not json at all"

    def test_an_error_is_always_returned_as_text_even_if_it_looks_like_json(self) -> None:
        assert agent.decode_result(text_result("[1, 2]", is_error=True)) == "[1, 2]"

    def test_structured_content_wrapping_a_list_is_unwrapped(self) -> None:
        # mcp 2.x servers that return a bare list publish it as {"result": [...]}
        assert agent.decode_result(text_result(structured={"result": [1, 2, 3]})) == [1, 2, 3]

    def test_structured_content_dict_is_returned_as_is(self) -> None:
        assert agent.decode_result(text_result('{"x": 1}', structured={"status": "sent"})) == {"status": "sent"}

    def test_a_result_without_content_decodes_to_empty_text(self) -> None:
        assert agent.decode_result(types.CallToolResult(content=[])) == ""
        assert agent.result_text(types.CallToolResult(content=[])) == ""

    def test_result_text_joins_text_blocks_and_falls_back_to_structured_json(self) -> None:
        assert agent.result_text(text_result("a", "b")) == "a\nb"
        assert agent.result_text(text_result(structured={"k": "v"})) == '{"k": "v"}'


class TestToolSignature:
    def test_required_and_optional_arguments(self) -> None:
        tool = types.Tool(
            name="demo",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}, "c": {"type": ["string", "null"]}},
                "required": ["a", "b"],
            },
        )
        assert agent.tool_signature(tool) == "demo(a: string, b: integer, c?: string|null)"

    def test_a_tool_without_arguments(self) -> None:
        assert agent.tool_signature(types.Tool(name="ping", input_schema={"type": "object"})) == "ping()"


class TestBuildCatalog:
    def test_indexes_tools_by_name_and_remembers_the_server(self) -> None:
        a = types.Tool(name="alpha", input_schema={"type": "object"})
        b = types.Tool(name="beta", input_schema={"type": "object"})
        catalog = agent.build_catalog({"s1": [a], "s2": [b]})
        assert {name: entry.server for name, entry in catalog.items()} == {"alpha": "s1", "beta": "s2"}

    def test_the_same_tool_name_on_two_servers_is_rejected(self) -> None:
        tool = types.Tool(name="search", input_schema={"type": "object"})
        with pytest.raises(ValueError, match="offered by both 's1' and 's2'"):
            agent.build_catalog({"s1": [tool], "s2": [tool]})


class TestTracer:
    def test_long_values_are_shortened_and_non_ascii_is_escaped(self) -> None:
        assert len(agent.compact_json({"k": "x" * 500}, 50)) == 50
        assert agent.compact_json({"k": "🚨"}, 50) == '{"k": "\\ud83d\\udea8"}'
        assert agent.clip("a\n  b", 10) == "a b"

    def test_first_sentence_stops_at_the_first_full_stop(self) -> None:
        assert agent.first_sentence("Does X. Returns Y and Z.", 100) == "Does X."
        assert agent.first_sentence("no stop here", 100) == "no stop here"
