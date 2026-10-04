"""Test fixture: a tiny MCP server for checking how the host behaves in unusual situations.

env_names  reports which environment variables the server process can see (names only)
crash      kills the server process in the middle of a call
"""

from __future__ import annotations

import json
import os

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("probe", version="0.0.0", log_level="WARNING")


@mcp.tool(description="List the names (never the values) of this process's environment variables.", structured_output=False)
def env_names() -> str:
    return json.dumps(sorted(os.environ))


@mcp.tool(description="Terminate the server process immediately, without replying.", structured_output=False)
def crash() -> str:
    os._exit(3)


@mcp.tool(description="Reply normally.", structured_output=False)
def ping() -> str:
    return "pong"


if __name__ == "__main__":
    mcp.run()
