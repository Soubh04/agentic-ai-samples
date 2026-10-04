"""Email MCP server (stdio transport).

Exposes one tool, ``send_email``. This is a demo: it NEVER sends real email. Every
message is appended as one JSON line to a local outbox file and the tool reports
``{"status": "sent", "message_id": ...}`` so the agent can carry on as if it had been
delivered.

The outbox location comes from the ``EMAIL_OUTBOX_PATH`` environment variable and
defaults to ``outbox.jsonl`` in the project folder (listed in ``.gitignore``).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

OUTBOX_ENV_VAR = "EMAIL_OUTBOX_PATH"
DEFAULT_OUTBOX = Path(__file__).resolve().parent.parent / "outbox.jsonl"


def outbox_path() -> Path:
    """Where messages are written: ``$EMAIL_OUTBOX_PATH`` or the project's ``outbox.jsonl``."""
    configured = os.environ.get(OUTBOX_ENV_VAR)
    return Path(configured).expanduser() if configured else DEFAULT_OUTBOX


def store_message(to: str, subject: str, body: str) -> str:
    """Append the message to the outbox and return its message id.

    Values are stored exactly as received (no trimming, no HTML escaping). The file is
    UTF-8 and non-ASCII characters are written as-is rather than as \\uXXXX escapes.
    """
    if not to.strip():
        raise ValueError("'to' must not be empty. Use an address or a list name such as 'ops-team'.")
    if not subject.strip():
        raise ValueError("'subject' must not be empty.")

    message_id = f"msg-{uuid.uuid4().hex[:12]}"
    record = {
        "message_id": message_id,
        "sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": to,
        "subject": subject,
        "body": body,
    }
    path = outbox_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the bytes identical on every platform (no CRLF translation).
    with path.open("a", encoding="utf-8", newline="\n") as outbox:
        outbox.write(json.dumps(record, ensure_ascii=False) + "\n")
    return message_id


mcp = MCPServer("email", version="0.1.0", log_level="WARNING")


@mcp.tool(
    description=(
        "Send an email to a person or a distribution list such as 'ops-team'. "
        "Returns {status, message_id}. Demo server: nothing is delivered, "
        "the message is appended to a local outbox file."
    ),
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False),
    structured_output=False,
)
def send_email(
    to: Annotated[str, Field(description="Recipient: an email address or a list name such as 'ops-team'.")],
    subject: Annotated[str, Field(description="Subject line.")],
    body: Annotated[str, Field(description="Plain-text message body.")],
) -> str:
    try:
        message_id = store_message(to, subject, body)
    except ValueError as exc:
        # In mcp 2.x only ToolError reaches the client with its message.
        raise ToolError(str(exc)) from exc
    return json.dumps({"status": "sent", "message_id": message_id})


if __name__ == "__main__":
    mcp.run()  # stdio transport
