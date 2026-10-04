"""Incidents MCP server (stdio transport).

Exposes one tool, ``search_incidents``, backed by a small in-memory dataset so the
results are identical on every run. All records are fictional.

The server speaks MCP over stdin/stdout, so when started by hand it looks idle.
Normally the agent (``agent.py``) starts it as a subprocess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

PRIORITIES = ("P1", "P2", "P3", "P4")
MIN_DAYS = 1
MAX_DAYS = 90

# The dataset is anchored to a fixed "now" so that "the last 7 days" returns the same
# incidents no matter when the sample is run.
REFERENCE_NOW = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Incident:
    id: str
    hours_ago: int
    priority: str
    service: str
    status: str
    title: str

    @property
    def opened_at(self) -> datetime:
        return REFERENCE_NOW - timedelta(hours=self.hours_ago)

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority,
            "service": self.service,
            "status": self.status,
            "opened_at": self.opened_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# Newest first. There are exactly 12 P1 incidents within the last 7 days (168 hours),
# plus older P1 incidents and a mix of P2-P4 to make the filters meaningful.
# No P4 incident falls inside the last 24 hours, so (P4, 1 day) is an empty result.
INCIDENTS: tuple[Incident, ...] = (
    Incident("INC-4821", 3, "P1", "checkout-api", "mitigated", "Elevated 5xx rate on checkout-api after config rollout"),
    Incident("INC-4820", 5, "P3", "notification-worker", "resolved", "Delayed push notifications for a subset of users"),
    Incident("INC-4818", 9, "P1", "payments-gateway", "resolved", "Card authorisation timeouts for one acquirer"),
    Incident("INC-4815", 14, "P2", "search-service", "mitigated", "Search latency p95 above 2 seconds"),
    Incident("INC-4813", 22, "P1", "search-service", "resolved", "Search index lag exceeding 20 minutes"),
    Incident("INC-4810", 30, "P4", "reporting-etl", "open", "Cosmetic misalignment in weekly report export"),
    Incident("INC-4808", 36, "P2", "auth-service", "resolved", "Password reset emails delayed"),
    Incident("INC-4806", 47, "P1", "auth-service", "resolved", "Intermittent login failures for SSO users"),
    Incident("INC-4803", 52, "P3", "inventory-sync", "resolved", "Duplicate SKUs created by import job"),
    Incident("INC-4801", 58, "P1", "orders-db", "resolved", "Primary replica failover during maintenance window"),
    Incident("INC-4798", 66, "P2", "checkout-api", "resolved", "Coupon validation errors for expired codes"),
    Incident("INC-4796", 70, "P1", "notification-worker", "resolved", "Notification queue backlog above 50k messages"),
    Incident("INC-4793", 79, "P1", "checkout-api", "resolved", "Cart totals mismatch for discounted baskets"),
    Incident("INC-4790", 88, "P3", "cdn-edge", "resolved", "Slow image loads in one edge region"),
    Incident("INC-4788", 93, "P1", "cdn-edge", "resolved", "Static asset 404s after cache purge"),
    Incident("INC-4785", 101, "P4", "user-profile-api", "open", "Avatar upload shows wrong file-size message"),
    Incident("INC-4783", 110, "P1", "inventory-sync", "resolved", "Stock levels stale after sync job crash"),
    Incident("INC-4780", 118, "P2", "payments-gateway", "resolved", "Webhook retries exceeding rate limit"),
    Incident("INC-4777", 127, "P1", "user-profile-api", "resolved", "Profile updates returning HTTP 500"),
    Incident("INC-4774", 139, "P3", "search-service", "resolved", "Autocomplete returning duplicate suggestions"),
    Incident("INC-4771", 146, "P1", "payments-gateway", "resolved", "Refund requests stuck in pending state"),
    Incident("INC-4768", 157, "P1", "reporting-etl", "resolved", "Nightly ETL failed; dashboards show stale data"),
    Incident("INC-4766", 165, "P2", "orders-db", "resolved", "Slow queries on order history endpoint"),
    # --- older than 7 days from here on ---
    Incident("INC-4759", 190, "P1", "checkout-api", "resolved", "Checkout page not loading on mobile web"),
    Incident("INC-4752", 214, "P3", "auth-service", "resolved", "MFA prompt shown twice"),
    Incident("INC-4744", 262, "P4", "cdn-edge", "resolved", "Incorrect cache headers on favicon"),
    Incident("INC-4731", 340, "P1", "auth-service", "resolved", "Token validation outage"),
    Incident("INC-4719", 420, "P2", "inventory-sync", "resolved", "Warehouse feed delayed by 3 hours"),
    Incident("INC-4702", 700, "P1", "orders-db", "resolved", "Storage saturation on orders database"),
    Incident("INC-4688", 760, "P3", "reporting-etl", "resolved", "Monthly export missing one region"),
    Incident("INC-4650", 2400, "P1", "payments-gateway", "resolved", "Certificate expiry broke settlement file upload"),
)


def _clip(value: str, limit: int = 40) -> str:
    """Shorten user-supplied text before echoing it back in an error message."""
    return value if len(value) <= limit else value[: limit - 3] + "..."


def find_incidents(priority: str, days: int) -> list[dict[str, str]]:
    """Return incidents of ``priority`` opened within the last ``days`` days, newest first.

    Raises ``ValueError`` with a message that is safe to show to the caller.
    """
    if priority not in PRIORITIES:
        raise ValueError(f"Invalid priority {_clip(priority)!r}: must be one of {', '.join(PRIORITIES)}.")
    if not MIN_DAYS <= days <= MAX_DAYS:
        raise ValueError(f"Invalid days {days}: must be a whole number between {MIN_DAYS} and {MAX_DAYS}.")
    cutoff = REFERENCE_NOW - timedelta(days=days)
    return [i.as_dict() for i in INCIDENTS if i.priority == priority and i.opened_at >= cutoff]


mcp = MCPServer("incidents", version="0.1.0", log_level="WARNING")


@mcp.tool(
    description=(
        "Search incidents of one priority (P1-P4) opened within the last N days (1-90). "
        "Returns a JSON array of incidents (id, title, priority, service, status, opened_at), newest first."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    structured_output=False,
)
def search_incidents(
    priority: Annotated[str, Field(description="Incident priority: P1 (most severe) to P4 (least severe).")],
    days: Annotated[int, Field(description="Look-back window in days, from 1 to 90.")],
) -> str:
    try:
        incidents = find_incidents(priority, days)
    except ValueError as exc:
        # In mcp 2.x only ToolError reaches the client with its message. Any other exception
        # is reported as a bare "Error executing tool" (the text stays on the server).
        raise ToolError(str(exc)) from exc
    return json.dumps(incidents)


if __name__ == "__main__":
    mcp.run()  # stdio transport
