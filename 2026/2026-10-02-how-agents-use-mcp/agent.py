"""A small AI agent that acts as an MCP *host*.

The agent starts three MCP servers as subprocesses (Incidents, Email, Docs), talks to
them over stdio, and prints a numbered trace that follows the eight steps of the post:

  Step 1  user -> agent            request      "Summarise this week's P1 incidents and email the team."
  Step 2  agent -> every server    tools/list   ask each server what it offers
  Step 3  every server -> agent    result       tool names + input schemas
  Step 4  agent -> incidents       tools/call   search_incidents {"priority": "P1", "days": 7}
  Step 5  incidents -> agent       result       12 incidents as JSON
  Step 6  agent -> email           tools/call   send_email {"to": "ops-team", ...}
  Step 7  email -> agent           result       {"status": "sent", "message_id": ...}
  Step 8  agent -> user            answer

Who decides which tool to call is the *planner*:

  --planner rules   (default) plain keyword rules; no network, no API key
  --planner claude  the Anthropic Messages API; Claude picks tools from the MCP tool list

Run it:  uv run python agent.py ["your request"] [--planner rules|claude]
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import anthropic
from mcp import ClientSession, MCPError, StdioServerParameters, stdio_client, types

PROJECT_ROOT = Path(__file__).resolve().parent
SERVERS: dict[str, Path] = {
    "incidents": PROJECT_ROOT / "servers" / "incidents_server.py",
    "email": PROJECT_ROOT / "servers" / "email_server.py",
    "docs": PROJECT_ROOT / "servers" / "docs_server.py",
}
SAMPLE_REQUEST = "Summarise this week's P1 incidents and email the team."
CLIENT_INFO = types.Implementation(name="how-agents-use-mcp", version="0.1.0")

# The MCP stdio client starts each server with a small allow-list of environment variables
# (PATH, HOME, ...), NOT with a copy of this process's environment. Only the variables
# named here are passed on. In particular ANTHROPIC_API_KEY never reaches a server.
FORWARDED_ENV = ("EMAIL_OUTBOX_PATH",)

DEFAULT_MODEL = "claude-opus-5-5"  # override with the ANTHROPIC_MODEL environment variable
MAX_TURNS = 8  # upper bound on model <-> tool round trips for one request


class AgentError(Exception):
    """An expected failure. The command line shows its message instead of a traceback."""


class PlanningError(AgentError):
    """The planner could not turn the request into tool calls."""


class ServerError(AgentError):
    """An MCP server would not start, or stopped answering."""


# --------------------------------------------------------------------------------------
# Data carried around by the host
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredTool:
    server: str
    tool: types.Tool


@dataclass
class ToolCall:
    """One ``tools/call`` round trip, as seen by the host."""

    server: str
    name: str
    arguments: dict[str, Any]
    is_error: bool
    result: Any  # parsed JSON payload when the server returned JSON, else plain text
    text: str  # the raw text the server returned; this is what an LLM gets to read


@dataclass
class AgentRun:
    request: str
    planner: str
    discovered: dict[str, list[str]]  # server name -> tool names found with tools/list
    calls: list[ToolCall]  # every tools/call, in order
    answer: str


def result_text(result: types.CallToolResult) -> str:
    """The text of a tool result (its text content blocks, or the structured content as JSON)."""
    parts = [block.text for block in result.content if isinstance(block, types.TextContent)]
    if parts:
        return "\n".join(parts)
    if result.structured_content is not None:
        return json.dumps(result.structured_content)
    return ""


def decode_result(result: types.CallToolResult) -> Any:
    """The payload of a tool result: parsed JSON where possible, otherwise text.

    Our servers return one JSON text block. Servers that return ``structured_content``
    instead are handled too (mcp 2.x wraps a bare list as ``{"result": [...]}``).
    """
    if result.is_error:
        return result_text(result)
    structured = result.structured_content
    if structured is not None:
        return structured["result"] if isinstance(structured, dict) and list(structured) == ["result"] else structured
    text = result_text(result)
    try:
        return json.loads(text)
    except ValueError:
        return text


def tool_signature(tool: types.Tool) -> str:
    """``name(arg: type, optional?: type)`` built from the tool's JSON Schema."""
    properties = tool.input_schema.get("properties", {})
    required = set(tool.input_schema.get("required", []))
    args = []
    for arg_name, spec in properties.items():
        kind = spec.get("type", "any")
        kind = "|".join(kind) if isinstance(kind, list) else str(kind)
        args.append(f"{arg_name}{'' if arg_name in required else '?'}: {kind}")
    return f"{tool.name}({', '.join(args)})"


def build_catalog(listings: Mapping[str, Sequence[types.Tool]]) -> dict[str, DiscoveredTool]:
    """Index every discovered tool by name. Tool names must be unique across servers.

    A production host would namespace them (``server__tool``). Here a clash is an error.
    """
    catalog: dict[str, DiscoveredTool] = {}
    for server, tools in listings.items():
        for tool in tools:
            if tool.name in catalog:
                raise ValueError(f"tool name {tool.name!r} is offered by both {catalog[tool.name].server!r} and {server!r}")
            catalog[tool.name] = DiscoveredTool(server, tool)
    return catalog


# --------------------------------------------------------------------------------------
# Trace output
# --------------------------------------------------------------------------------------


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compact_json(value: Any, limit: int) -> str:
    """One-line JSON for the trace. ASCII-escaped, so it prints on any terminal."""
    return clip(json.dumps(value, ensure_ascii=True), limit)


def first_sentence(text: str, limit: int) -> str:
    text = " ".join(text.split())
    head, dot, _ = text.partition(". ")
    return clip(head + dot.strip(), limit)


def preview_item(item: Any, limit: int = 110) -> str:
    """One line for a list item: its first three fields as key=value, or compact JSON."""
    if isinstance(item, dict):
        return clip(" | ".join(f"{key}={value}" for key, value in list(item.items())[:3]), limit)
    return compact_json(item, limit)


class Tracer:
    """Prints what the user, the host and the servers say to each other, one step per message."""

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out
        self.step = 0

    def _print(self, text: str = "") -> None:
        print(text, file=self._out if self._out is not None else sys.stdout, flush=True)

    def _step(self, direction: str, kind: str, detail: str = "", *more: str) -> None:
        self.step += 1
        label = f"Step {self.step}"
        self._print(f"{label:<8} {direction:<22} {kind:<11} {detail}".rstrip())
        for line in more:
            self._print(f"{'':<8} {line}")

    def note(self, text: str) -> None:
        self._print(f"{'':<8} {text}")

    def llm(self, direction: str, detail: str) -> None:
        """A message to or from the model API. Not an MCP message, so it gets no step number."""
        self._print(f"{'LLM':<8} {direction:<22} {detail}")

    def user_request(self, text: str) -> None:
        self._step("user -> agent", "request", text)

    def connected(self, server: str, init: types.InitializeResult) -> None:
        info = init.server_info
        self.note(f"connect: {server}  initialize ok (protocol {init.protocol_version}, server {info.name} {info.version})")

    def tools_list_request(self, servers: Sequence[str]) -> None:
        self._step("agent -> every server", "tools/list", ", ".join(servers))

    def tools_list_result(self, listings: Mapping[str, Sequence[types.Tool]]) -> None:
        lines = []
        for server, tools in listings.items():
            for tool in tools:
                hint = tool.annotations.read_only_hint if tool.annotations else None
                tag = "" if hint is None else "  [read-only]" if hint else "  [not read-only]"
                lines.append(f"  {server:<10} {tool_signature(tool)}{tag}")
                lines.append(f"  {'':<10} {first_sentence(tool.description or '', 110)}")
        self._step("every server -> agent", "result", "tool names + input schemas", *lines)

    def tool_call(self, server: str, name: str, arguments: Mapping[str, Any]) -> None:
        self._step(f"agent -> {server}", "tools/call", f"{name} {compact_json(arguments, 110)}")

    def tool_result(self, call: ToolCall) -> None:
        direction = f"{call.server} -> agent"
        if call.is_error:
            self._step(direction, "result", f"isError=true: {clip(call.text, 140)}")
        elif isinstance(call.result, list):
            n = len(call.result)
            preview = [f"  {preview_item(item)}" for item in call.result[:2]]
            if n > 2:
                preview.append(f"  ... and {n - 2} more")
            self._step(direction, "result", f"{n} item{'s' if n != 1 else ''} (JSON array)", *preview)
        else:
            self._step(direction, "result", compact_json(call.result, 140))

    def answer(self, text: str) -> None:
        first, *rest = text.strip().splitlines() or [""]
        self._step("agent -> user", "answer", first, *rest)


# --------------------------------------------------------------------------------------
# The host: one MCP client session per server
# --------------------------------------------------------------------------------------


class Host:
    """Owns the connections to the MCP servers, remembers what they offer, routes tool calls."""

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.sessions: dict[str, ClientSession] = {}
        self.catalog: dict[str, DiscoveredTool] = {}
        self.calls: list[ToolCall] = []

    async def connect(self, stack: AsyncExitStack, servers: Mapping[str, Path] = SERVERS) -> None:
        """Start every server as a subprocess and run the MCP ``initialize`` handshake."""
        for name, script in servers.items():
            params = StdioServerParameters(
                command=sys.executable,  # same interpreter and virtualenv as this process
                args=[str(script)],
                env={key: os.environ[key] for key in FORWARDED_ENV if key in os.environ},
            )
            # stderr of the server is passed through, so a crashing server shows its traceback.
            errlog = sys.__stderr__ or sys.stderr  # the real stderr, even when a test runner captures sys.stderr
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params, errlog=errlog))
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream, read_timeout_seconds=30, client_info=CLIENT_INFO)
            )
            try:
                init = await session.initialize()  # sends `initialize`, then `notifications/initialized`
            except MCPError as exc:
                raise ServerError(
                    f"MCP server {name!r} did not complete the initialize handshake ({exc}). "
                    f"Any error output from the server is shown above; try running 'python {script}' directly."
                ) from exc
            self.sessions[name] = session
            self.tracer.connected(name, init)

    async def discover(self) -> None:
        """Send ``tools/list`` to every server and build the tool catalog from the answers."""
        self.tracer.tools_list_request(list(self.sessions))
        listings: dict[str, list[types.Tool]] = {}
        for name, session in self.sessions.items():
            try:
                # One page is enough for these servers. A production host follows `next_cursor`.
                listings[name] = (await session.list_tools()).tools
            except MCPError as exc:
                raise ServerError(f"MCP server {name!r} failed to answer tools/list ({exc})") from exc
        try:
            self.catalog = build_catalog(listings)
        except ValueError as exc:  # the same tool name on two servers
            raise ServerError(str(exc)) from exc
        self.tracer.tools_list_result(listings)

    def require(self, tool_name: str) -> None:
        if tool_name not in self.catalog:
            offered = ", ".join(self.catalog) or "nothing"
            raise PlanningError(f"no connected MCP server offers the tool {tool_name!r} (discovered: {offered})")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Send ``tools/call`` to the server that owns ``name`` and wrap the answer.

        A failed call (bad arguments, server-side error) comes back as ``is_error=True``
        instead of an exception, so a planner can show the message or let the model retry.
        """
        entry = self.catalog.get(name)
        if entry is None:
            message = f"Unknown tool {name!r}. Available tools: {', '.join(self.catalog)}."
            self.tracer.note(f"(model asked for unknown tool {name!r}; nothing was sent to any server)")
            return ToolCall("-", name, arguments, True, message, message)

        self.tracer.tool_call(entry.server, name, arguments)
        try:
            result = await self.sessions[entry.server].call_tool(name, arguments)
            call = ToolCall(entry.server, name, arguments, result.is_error, decode_result(result), result_text(result))
        except MCPError as exc:  # a JSON-RPC protocol error, as opposed to a tool reporting isError
            message = f"MCP protocol error: {exc}"
            call = ToolCall(entry.server, name, arguments, True, message, message)
        self.tracer.tool_result(call)
        self.calls.append(call)
        return call


# --------------------------------------------------------------------------------------
# Planner 1: keyword rules (no network, no API key)
# --------------------------------------------------------------------------------------

# (pattern matched against the request, tool the rule needs). Tools that change things
# (email) come last so they run after the tools that gather information.
RULES: tuple[tuple[str, str], ...] = (
    (r"\bincidents?\b", "search_incidents"),
    (r"\b(?:runbooks?|docs?|documentation|playbooks?)\b", "search_docs"),
    (r"\b(?:e-?mails?|mail|notify|send)\b", "send_email"),
)


def extract_priority(request: str) -> str:
    """'P1' ... 'P4' from the request. Defaults to P1; an unknown level such as P9 is passed on as-is."""
    match = re.search(r"\bp(\d+)\b", request, re.IGNORECASE)
    return f"P{match.group(1)}" if match else "P1"


def extract_days(request: str) -> int:
    """Look-back window in days: 'last 3 days', 'today', 'this week', 'this month'. Defaults to 7."""
    text = request.lower()
    if match := re.search(r"\b(?:last|past)\s+(\d{1,4})\s+days?\b", text):
        return int(match.group(1))
    if re.search(r"\b(?:today|24\s*hours?)\b", text):
        return 1
    if re.search(r"\bmonth\b", text):
        return 30
    return 7  # "this week", "last week", or nothing at all


def extract_recipient(request: str) -> str:
    """An email address from the request, otherwise the 'ops-team' list."""
    match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", request)
    return match.group(0) if match else "ops-team"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def compose_email(
    priority: str, days: int, incidents: list[dict[str, Any]] | None, links: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """Plain-text subject and body summarising whatever the search tools returned."""
    window = f"last {_count(days, 'day')}"
    lines = ["Hi team,", ""]
    if incidents is None:
        subject = "Runbook links"
    else:
        subject = f"{priority} incident summary - {window} ({_count(len(incidents), 'incident')})"
        if not incidents:
            lines += [f"No {priority} incidents were opened in the {window}."]
        else:
            lines += [f"{_count(len(incidents), priority + ' incident')} opened in the {window}.", "", "By service:"]
            by_service = Counter(i["service"] for i in incidents)
            lines += [f"- {service}: {n}" for service, n in sorted(by_service.items(), key=lambda kv: (-kv[1], kv[0]))]
            lines += ["", "Incidents, newest first:"]
            lines += [f"- {i['id']} [{i['status']}] {i['service']}: {i['title']}" for i in incidents]
    if links:
        lines += ["", "Related runbooks:"] + [f"- {link['title']}: {link['url']}" for link in links]
    lines += ["", "Regards,", "Incident assistant (demo)"]
    return subject, "\n".join(lines)


async def rules_planner(host: Host, request: str) -> str:
    """Pick tools with fixed keyword rules. The same request always gives the same plan."""
    wanted = [tool for pattern, tool in RULES if re.search(pattern, request, re.IGNORECASE)]
    if not set(wanted) & {"search_incidents", "search_docs"}:
        raise PlanningError(
            "the rules planner can search incidents or runbooks and email a summary of the results; "
            "this request mentions neither. Rephrase it, or use --planner claude for free-form requests."
        )
    for tool in wanted:
        host.require(tool)

    priority, days = extract_priority(request), extract_days(request)
    incidents = links = None
    answer: list[str] = []

    if "search_incidents" in wanted:
        call = await host.call_tool("search_incidents", {"priority": priority, "days": days})
        if call.is_error:
            return f"search_incidents failed: {call.text}"
        incidents = call.result
        answer.append(f"Found {_count(len(incidents), priority + ' incident')} in the last {_count(days, 'day')}.")

    if "search_docs" in wanted:
        call = await host.call_tool("search_docs", {"query": request})
        if call.is_error:
            return f"search_docs failed: {call.text}"
        links = call.result
        bullets = "".join(f"\n- {link['title']}: {link['url']}" for link in links)
        answer.append(f"Found {_count(len(links), 'runbook')}.{bullets}")

    if "send_email" in wanted:
        recipient = extract_recipient(request)
        subject, body = compose_email(priority, days, incidents, links)
        call = await host.call_tool("send_email", {"to": recipient, "subject": subject, "body": body})
        if call.is_error:
            return f"send_email failed: {call.text}"
        receipt = call.result
        answer.append(f"Emailed a summary to {recipient} (message {receipt['message_id']}, status: {receipt['status']}).")

    return "\n".join(answer)


# --------------------------------------------------------------------------------------
# Planner 2: Claude chooses the tools (Anthropic Messages API with tool use)
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an on-call operations assistant. You can act only through the tools you are given. "
    "Use them to complete the user's request, then reply with a short plain-text summary of what you did. "
    "If a tool returns an error, read the message and retry with corrected arguments."
)

_CLAUDE_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


def mcp_tool_to_claude_tool(tool: types.Tool) -> dict[str, Any]:
    """Convert an MCP tool definition into a Claude Messages API tool definition.

        MCP    {name, description, inputSchema}    (``tool.input_schema`` in the Python SDK)
        Claude {name, description, input_schema}

    Both describe the arguments with JSON Schema, so the schema is passed through as it is
    (copied, not shared). The only check is the name: Claude accepts letters, digits,
    '_' and '-', up to 64 characters.
    """
    if not _CLAUDE_TOOL_NAME.fullmatch(tool.name):
        raise ValueError(
            f"MCP tool name {tool.name!r} is not a valid Claude tool name "
            "(letters, digits, '_' and '-', up to 64 characters)"
        )
    schema = copy.deepcopy(tool.input_schema)
    schema.setdefault("type", "object")  # Claude requires an object schema at the root
    definition: dict[str, Any] = {"name": tool.name}
    if tool.description:
        definition["description"] = tool.description
    definition["input_schema"] = schema
    return definition


def build_claude_request(model: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.create(...)``."""
    return {
        "model": model,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "tools": tools,
        "messages": messages,
        # No `temperature` and no forced `tool_choice`: current models (claude-opus-5-5, claude-sonnet-5-5) reject both.
        # The default tool_choice ("auto") lets the model decide whether and which tool to call.
        "output_config": {"effort": "medium"},
        # Opt in to Anthropic's server-side refusal fallback for this model family. To use the
        # plain endpoint instead, delete the next two entries and call client.messages.create.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


async def claude_planner(
    host: Host,
    request: str,
    *,
    client: anthropic.AsyncAnthropic | None = None,
    model: str | None = None,
) -> str:
    """Let Claude choose the tools. The host stays in charge: it makes every MCP call itself.

    The loop: send the request and the converted MCP tools -> if Claude answers with
    ``tool_use`` blocks, run each one as an MCP ``tools/call`` and send the results back
    as ``tool_result`` blocks -> repeat until Claude replies with plain text.
    """
    client = client or anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment
    model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    tools = [mcp_tool_to_claude_tool(entry.tool) for entry in host.catalog.values()]
    messages: list[dict[str, Any]] = [{"role": "user", "content": request}]

    for _ in range(MAX_TURNS):
        host.tracer.llm(
            "agent -> Claude API",
            f"messages.create: model {model}, {_count(len(tools), 'tool')}, {_count(len(messages), 'message')}",
        )
        response = await client.beta.messages.create(**build_claude_request(model, tools, messages))
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        chosen = ", ".join(block.name for block in tool_uses) or "none"
        host.tracer.llm(
            "Claude API -> agent",
            f"stop_reason={response.stop_reason}, tools chosen: {chosen} "
            f"(tokens in/out {response.usage.input_tokens}/{response.usage.output_tokens})",
        )

        if tool_uses:  # remarks the model makes while it works; the final reply is shown as the answer
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    host.tracer.llm("Claude API -> agent", f'says: "{clip(block.text, 100)}"')

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            return f"The model declined this request (category: {category or 'unspecified'})."
        if response.stop_reason == "max_tokens":
            raise PlanningError("the model reply was cut off at max_tokens")

        # Keep the whole assistant turn, including any thinking blocks, exactly as received.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use" or not tool_uses:
            return "\n".join(block.text for block in response.content if block.type == "text").strip()

        results = []
        for block in tool_uses:  # all tool results go back together, in ONE user message
            call = await host.call_tool(block.name, dict(block.input))
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id, "content": call.text or "(no output)"}
            if call.is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})

    raise PlanningError(f"the model was still calling tools after {MAX_TURNS} turns; giving up")


# --------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------


async def run_agent(
    request: str = SAMPLE_REQUEST,
    planner: str = "rules",
    *,
    tracer: Tracer | None = None,
    claude_client: anthropic.AsyncAnthropic | None = None,
    servers: Mapping[str, Path] | None = None,
) -> AgentRun:
    """Connect to the servers, discover their tools, let the planner act, return what happened.

    ``servers`` maps a name to the script of an MCP server; the default is the three in ``servers/``.
    """
    if planner not in ("rules", "claude"):
        raise ValueError(f"unknown planner {planner!r}; expected 'rules' or 'claude'")
    tracer = tracer or Tracer()
    host = Host(tracer)
    tracer.user_request(request)

    failure: Exception | None = None
    answer = ""
    async with AsyncExitStack() as stack:
        try:
            await host.connect(stack, servers or SERVERS)
            await host.discover()
            if planner == "claude":
                answer = await claude_planner(host, request, client=claude_client)
            else:
                answer = await rules_planner(host, request)
        except (AgentError, anthropic.APIError) as exc:
            # Expected failures are held here and re-raised below, after the server
            # connections have closed. An exception that propagates out of the MCP client
            # contexts arrives wrapped in (nested) ExceptionGroups.
            failure = exc
    if failure is not None:
        raise failure

    tracer.answer(answer)
    discovered = {
        server: [e.tool.name for e in host.catalog.values() if e.server == server] for server in host.sessions
    }
    return AgentRun(request, planner, discovered, host.calls, answer)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A small MCP host: discovers tools on three MCP servers and uses them to answer a request.",
    )
    parser.add_argument("request", nargs="?", default=SAMPLE_REQUEST, help=f'what to ask (default: "{SAMPLE_REQUEST}")')
    parser.add_argument(
        "--planner",
        choices=["rules", "claude"],
        default="rules",
        help=(
            "rules: deterministic keyword rules, no network (default). "
            "claude: Claude chooses the tools; needs ANTHROPIC_API_KEY"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.planner == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: --planner claude needs an Anthropic API key. Export ANTHROPIC_API_KEY in your shell "
            "(never put it in code or commit it), or run without --planner to use the offline rules planner.",
            file=sys.stderr,
        )
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # an unusual character in a reply must not crash the trace

    try:
        asyncio.run(run_agent(args.request, args.planner))
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("error: the Anthropic API rejected the API key (HTTP 401). Check ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"error: could not reach the Anthropic API: {exc}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"error: the Anthropic API returned HTTP {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIError as exc:
        print(f"error: Anthropic API error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
