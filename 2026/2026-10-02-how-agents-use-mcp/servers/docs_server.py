"""Docs MCP server (stdio transport).

Exposes one tool, ``search_docs``, that looks up runbooks in a tiny fictional library.
All titles and URLs are made up (``docs.example.com`` is a reserved example domain).

In the sample request the agent discovers this server but never calls it: the request
is about incidents and email, not about documentation.
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

MAX_RESULTS = 5

# (title, url, extra keywords)
RUNBOOKS: tuple[tuple[str, str, str], ...] = (
    ("Checkout API: elevated 5xx error rate", "https://docs.example.com/runbooks/checkout-api-5xx",
     "checkout api http errors rollout config"),
    ("Payments gateway: authorisation timeouts", "https://docs.example.com/runbooks/payments-gateway-timeouts",
     "payments gateway card acquirer timeout latency"),
    ("Search service: index lag", "https://docs.example.com/runbooks/search-index-lag",
     "search index lag latency stale"),
    ("Auth service: login failures", "https://docs.example.com/runbooks/auth-login-failures",
     "auth login sso token sign-in failures"),
    ("Orders database: replica failover", "https://docs.example.com/runbooks/orders-db-failover",
     "orders database db replica failover storage"),
    ("Notification worker: queue backlog", "https://docs.example.com/runbooks/notification-queue-backlog",
     "notification worker queue backlog messages"),
    ("CDN edge: asset 404s and cache purge", "https://docs.example.com/runbooks/cdn-edge-404s",
     "cdn edge cache purge static assets"),
    ("Incident response: severity definitions", "https://docs.example.com/runbooks/incident-severity",
     "incident response severity priority p1 p2 p3 p4 definitions"),
)


def _words(text: str) -> set[str]:
    """Lower-case words with a naive plural fold, so 'errors' matches 'error'."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text.lower())}


def find_docs(query: str) -> list[dict[str, str]]:
    """Return up to ``MAX_RESULTS`` runbooks that share words with ``query``, best match first."""
    wanted = _words(query)
    if not wanted:
        raise ValueError("'query' must contain at least one word.")
    scored = []
    for title, url, keywords in RUNBOOKS:
        score = len(wanted & _words(f"{title} {keywords}"))
        if score:
            scored.append((-score, title, {"title": title, "url": url}))
    scored.sort(key=lambda item: item[:2])
    return [entry for _, _, entry in scored[:MAX_RESULTS]]


mcp = MCPServer("docs", version="0.1.0", log_level="WARNING")


@mcp.tool(
    description=(
        "Search the runbook library by keywords (for example a service name or symptom). "
        "Returns a JSON array of {title, url}, best match first; empty if nothing matches."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    structured_output=False,
)
def search_docs(
    query: Annotated[str, Field(description="Keywords to look for, e.g. 'checkout-api 5xx'.")],
) -> str:
    try:
        results = find_docs(query)
    except ValueError as exc:
        # In mcp 2.x only ToolError reaches the client with its message.
        raise ToolError(str(exc)) from exc
    return json.dumps(results)


if __name__ == "__main__":
    mcp.run()  # stdio transport
