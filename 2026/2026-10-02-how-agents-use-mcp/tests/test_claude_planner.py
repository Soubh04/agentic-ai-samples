"""The Claude planner, tested WITHOUT calling the real Claude API.

The Anthropic SDK is real; only its HTTP layer is replaced by canned replies (httpx2's
MockTransport). So the requests that would go to the API are built, serialized and
recorded exactly as in production, but nothing leaves the machine and no API key is needed.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from conftest import read_outbox, run_async
from jsonschema import Draft202012Validator
from mcp import types

import agent

# ---- the MCP -> Claude tool conversion -------------------------------------------------


def mcp_tool(name: str = "demo", description: str | None = "Does a demo.", **schema: Any) -> types.Tool:
    input_schema = schema or {
        "type": "object",
        "properties": {"q": {"type": "string", "description": "query"}},
        "required": ["q"],
    }
    return types.Tool(name=name, description=description, input_schema=input_schema)


def test_conversion_maps_name_description_and_schema() -> None:
    tool = mcp_tool()
    assert agent.mcp_tool_to_claude_tool(tool) == {
        "name": "demo",
        "description": "Does a demo.",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "query"}},
            "required": ["q"],
        },
    }


def test_conversion_keeps_the_whole_json_schema_including_nested_parts() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"enum": ["a", "b"]},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        },
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}},
        "required": ["items"],
        "additionalProperties": False,
    }
    converted = agent.mcp_tool_to_claude_tool(mcp_tool(**copy.deepcopy(schema)))
    assert converted["input_schema"] == schema


def test_conversion_copies_the_schema_instead_of_sharing_it() -> None:
    tool = mcp_tool()
    converted = agent.mcp_tool_to_claude_tool(tool)
    converted["input_schema"]["properties"]["q"]["type"] = "number"
    assert tool.input_schema["properties"]["q"]["type"] == "string"


@pytest.mark.parametrize("description", [None, ""])
def test_conversion_omits_a_missing_description(description: str | None) -> None:
    assert "description" not in agent.mcp_tool_to_claude_tool(mcp_tool(description=description))


def test_conversion_adds_the_object_type_when_the_schema_has_none() -> None:
    converted = agent.mcp_tool_to_claude_tool(mcp_tool(properties={}))
    assert converted["input_schema"]["type"] == "object"


@pytest.mark.parametrize("name", ["search incidents", "search.incidents", "naïve", "", "x" * 65, "a/b"])
def test_conversion_rejects_names_claude_would_refuse(name: str) -> None:
    with pytest.raises(ValueError, match="not a valid Claude tool name"):
        agent.mcp_tool_to_claude_tool(mcp_tool(name=name))


@pytest.mark.parametrize("name", ["search_incidents", "send-email", "A1", "x" * 64])
def test_conversion_accepts_valid_names(name: str) -> None:
    assert agent.mcp_tool_to_claude_tool(mcp_tool(name=name))["name"] == name


# ---- the request that goes to the API ---------------------------------------------------


def test_request_does_not_force_a_tool_or_set_sampling_parameters() -> None:
    request = agent.build_claude_request("claude-sonnet-5-5", [], [{"role": "user", "content": "hi"}])
    # Forced tool_choice and non-default sampling parameters are rejected by the Claude 5 family.
    for rejected in ("tool_choice", "temperature", "top_p", "top_k", "thinking"):
        assert rejected not in request
    assert request["model"] == "claude-sonnet-5-5"
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert request["max_tokens"] >= 4096


# ---- a fake Anthropic HTTP layer ----------------------------------------------------------


class FakeClaude:
    """Scripted replies for the Messages API. Records every request the SDK sends."""

    def __init__(self, *replies: dict[str, Any] | httpx2.Response) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.client = anthropic.AsyncAnthropic(
            api_key="test-key-not-real",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(
            {"path": request.url.path, "headers": dict(request.headers), "body": json.loads(request.content)}
        )
        assert self._replies, "the agent called the API more often than the test scripted"
        reply = self._replies.pop(0)
        return reply if isinstance(reply, httpx2.Response) else httpx2.Response(200, json=reply)


def reply(content: list[dict[str, Any]], stop_reason: str = "end_turn", **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": agent.DEFAULT_MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
        **extra,
    }


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(tool_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": arguments}


def run_claude(
    fake: FakeClaude, request_text: str = agent.SAMPLE_REQUEST, trace: io.StringIO | None = None
) -> agent.AgentRun:
    tracer = agent.Tracer(trace if trace is not None else io.StringIO())
    return run_async(agent.run_agent(request_text, "claude", tracer=tracer, claude_client=fake.client))


# ---- full runs: fake model, REAL MCP servers over stdio ------------------------------------


def test_claude_picks_the_tools_and_the_host_executes_them_over_mcp(outbox: Path) -> None:
    summary = "12 P1 incidents this week. Details to follow."
    fake = FakeClaude(
        reply(
            [
                {"type": "thinking", "thinking": "", "signature": "sig-1"},
                text("I'll start with the incidents."),
                tool_use("toolu_1", "search_incidents", priority="P1", days=7),
            ],
            "tool_use",
        ),
        reply([tool_use("toolu_2", "send_email", to="ops-team", subject="P1 weekly", body=summary)], "tool_use"),
        reply([text("Emailed the P1 summary to ops-team.")]),
    )

    trace = io.StringIO()
    run = run_claude(fake, trace=trace)

    # The host made the MCP calls the model asked for, in order.
    assert [(c.server, c.name) for c in run.calls] == [("incidents", "search_incidents"), ("email", "send_email")]
    assert len(run.calls[0].result) == 12
    assert run.calls[1].result["status"] == "sent"
    assert "search_docs" in run.discovered["docs"] and "search_docs" not in [c.name for c in run.calls]
    assert run.answer == "Emailed the P1 summary to ops-team."
    (stored,) = read_outbox(outbox)
    assert (stored["to"], stored["subject"], stored["body"]) == ("ops-team", "P1 weekly", summary)

    first, second, third = fake.requests
    assert first["path"] == "/v1/messages"

    # Request 1: the discovered MCP tools, converted, plus the user's request.
    body = first["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["messages"] == [{"role": "user", "content": agent.SAMPLE_REQUEST}]
    assert [t["name"] for t in body["tools"]] == ["search_incidents", "send_email", "search_docs"]
    for definition in body["tools"]:
        assert definition["description"]
        Draft202012Validator.check_schema(definition["input_schema"])  # a valid JSON Schema
        assert definition["input_schema"]["type"] == "object"
    assert body["tools"][0]["input_schema"]["required"] == ["priority", "days"]
    for rejected in ("tool_choice", "temperature", "thinking"):
        assert rejected not in body
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in first["headers"]["anthropic-beta"]
    assert body["output_config"] == {"effort": "medium"}

    # Request 2: the whole assistant turn is sent back unchanged (thinking block included),
    # followed by ONE user message holding the tool_result for the matching tool_use id.
    assistant, user = second["body"]["messages"][1:]
    assert [block["type"] for block in assistant["content"]] == ["thinking", "text", "tool_use"]
    assert assistant["content"][0]["signature"] == "sig-1"
    (tool_result,) = user["content"]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert "is_error" not in tool_result
    assert len(json.loads(tool_result["content"])) == 12

    # The trace labels model traffic separately from the numbered MCP steps.
    shown = trace.getvalue()
    assert "LLM      agent -> Claude API" in shown
    assert "stop_reason=tool_use, tools chosen: search_incidents" in shown
    assert 'says: "I\'ll start with the incidents."' in shown
    assert [line.split()[0] for line in shown.splitlines() if line.startswith("Step ")].count("Step") == 8

    # Request 3: the email receipt goes back the same way.
    receipt = third["body"]["messages"][-1]["content"][0]
    assert receipt["tool_use_id"] == "toolu_2" and json.loads(receipt["content"])["status"] == "sent"


def test_a_tool_error_goes_back_to_the_model_which_can_correct_itself(outbox: Path) -> None:
    fake = FakeClaude(
        reply([tool_use("toolu_1", "search_incidents", priority="P9", days=7)], "tool_use"),
        reply([tool_use("toolu_2", "search_incidents", priority="P1", days=7)], "tool_use"),
        reply([text("There were 12 P1 incidents.")]),
    )

    run = run_claude(fake, "How many P1 incidents this week?")

    assert [c.is_error for c in run.calls] == [True, False]
    assert len(run.calls[1].result) == 12
    assert run.answer == "There were 12 P1 incidents."
    (error_result,) = fake.requests[1]["body"]["messages"][-1]["content"]
    assert error_result["is_error"] is True
    assert "P9" in error_result["content"] and "P1, P2, P3, P4" in error_result["content"]
    assert not outbox.exists()


def test_an_api_error_surfaces_as_a_plain_anthropic_error_not_an_exception_group() -> None:
    unauthorized = httpx2.Response(
        401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    )
    with pytest.raises(anthropic.AuthenticationError):
        run_claude(FakeClaude(unauthorized))


# ---- the loop logic on its own: stub host, no servers ---------------------------------------


def stub_host(*tools: types.Tool) -> agent.Host:
    host = agent.Host(agent.Tracer(io.StringIO()))
    host.catalog = agent.build_catalog({"stub": list(tools or [mcp_tool(name="lookup")])})
    return host


def test_model_comes_from_anthropic_model_with_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for value, expected in [(None, "claude-opus-5-5"), ("claude-sonnet-5-5", "claude-sonnet-5-5")]:
        if value is None:
            monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        else:
            monkeypatch.setenv("ANTHROPIC_MODEL", value)
        fake = FakeClaude(reply([text("hello")]))
        assert run_async(agent.claude_planner(stub_host(), "hi", client=fake.client)) == "hello"
        assert fake.requests[0]["body"]["model"] == expected


def test_a_refusal_is_reported_instead_of_crashing() -> None:
    refusal = reply([], "refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "no"})
    answer = run_async(agent.claude_planner(stub_host(), "hi", client=FakeClaude(refusal).client))
    assert answer == "The model declined this request (category: cyber)."


def test_a_reply_cut_off_at_max_tokens_is_an_error() -> None:
    fake = FakeClaude(reply([text("partial")], "max_tokens"))
    with pytest.raises(agent.PlanningError, match="max_tokens"):
        run_async(agent.claude_planner(stub_host(), "hi", client=fake.client))


def test_a_tool_the_model_invented_is_answered_with_an_error_and_never_sent_to_a_server() -> None:
    fake = FakeClaude(
        reply([tool_use("toolu_1", "delete_everything", confirm=True)], "tool_use"),
        reply([text("I can't do that.")]),
    )
    host = stub_host()
    assert run_async(agent.claude_planner(host, "wipe it", client=fake.client)) == "I can't do that."
    assert host.calls == []  # nothing reached a server
    (result,) = fake.requests[1]["body"]["messages"][-1]["content"]
    assert result["is_error"] is True and "Unknown tool 'delete_everything'" in result["content"]


def test_several_tool_calls_in_one_turn_are_answered_together_in_one_user_message() -> None:
    fake = FakeClaude(
        reply([tool_use("t1", "lookup", q="a"), tool_use("t2", "lookup", q="b")], "tool_use"),
        reply([text("done")]),
    )
    host = stub_host()

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> agent.ToolCall:
        return agent.ToolCall("stub", name, arguments, False, arguments["q"], f"result-{arguments['q']}")

    host.call_tool = fake_call_tool  # type: ignore[method-assign]
    assert run_async(agent.claude_planner(host, "go", client=fake.client)) == "done"
    last = fake.requests[1]["body"]["messages"][-1]
    assert last["role"] == "user"
    assert [(r["tool_use_id"], r["content"]) for r in last["content"]] == [("t1", "result-a"), ("t2", "result-b")]


def test_a_model_that_never_stops_calling_tools_is_cut_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "MAX_TURNS", 2)
    fake = FakeClaude(*[reply([tool_use(f"t{i}", "nope")], "tool_use") for i in range(2)])
    with pytest.raises(agent.PlanningError, match="after 2 turns"):
        run_async(agent.claude_planner(stub_host(), "loop", client=fake.client))
    assert len(fake.requests) == 2
