"""Shared types, the rubric both backends receive, and the price table. No network, no SDKs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# --- The decision both backends make -----------------------------------------------------------
# One rubric, written once: Jev receives it as question criteria, Claude in its prompt.
TEAMS = {
    "billing": "Charges, invoices, payment methods, refunds and credits.",
    "technical": "Errors, outages, bugs, logins, integrations and performance.",
    "sales": "Pricing, quotes, upgrades, demos and new purchases.",
}
# Ordered from least to most urgent: a level's position is the score Jev returns for it (0, 1, 2).
URGENCY = {
    "can wait": "No deadline. A general question or feedback.",
    "this week": "A problem or deadline within days. The customer can keep working meanwhile.",
    "today": "The customer is blocked, money is at risk, or the deadline is today.",
}
REFUND = "The customer is asking for their money back (a refund or a credit)."

# --- Published list prices: the ONLY place a price appears -------------------------------------
# USD per million tokens, checked on 2026-10-02. Prices change: re-check before you quote them.
# Cost = input tokens x input price + output tokens x output price. No caching or batch discount.
PRICES_CHECKED = "2026-10-02"
PRICES_USD_PER_MTOK = {
    # https://docs.typesafe.ai/models : input $0.042 per MTok, "Output tokens are free".
    "typesafe-jev": {"input": 0.042, "output": 0.0},
    # https://platform.claude.com/docs/en/about-claude/pricing : Claude Opus 5.5 base rates.
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}


@dataclass(frozen=True)
class CallStats:
    """What one real API call measured. A mock backend never creates one."""

    model: str  # the model the API says served the call
    latency_ms: float  # wall-clock around the SDK call: network, server time and any SDK retries
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None  # None when tokens are unknown or no published price covers this model


@dataclass(frozen=True)
class Decision:
    team: str
    urgency: str
    refund_request: bool
    team_confidence: float | None = None  # Jev only: an LLM returns a label, not a probability
    team_probabilities: dict[str, float] | None = None  # Jev only
    refund_probability: float | None = None  # Jev only
    stats: CallStats | None = None  # None means nothing was measured (mock backend)


@dataclass(frozen=True)
class Draft:
    """A reply written for a human to review. It is never sent."""

    text: str
    stats: CallStats | None = None


class BackendError(RuntimeError):
    """A backend returned something unusable: truncated, malformed or missing an answer."""


class Declined(BackendError):
    """Claude declined the request (stop_reason "refusal"). `category` is Anthropic's label."""

    def __init__(self, category: str | None = None) -> None:
        super().__init__(f"Claude declined this request (category: {category or 'not given'})")
        self.category = category


@runtime_checkable
class Decider(Protocol):
    """Interface 1: ticket in, Decision out. Jev, Claude and both mocks implement it."""

    def decide(self, ticket: str) -> Decision: ...


@runtime_checkable
class Writer(Protocol):
    """Interface 2: ticket in, a reply draft out. Claude and its mock implement it."""

    def draft_reply(self, ticket: str) -> Draft: ...


def cost_of(price_key: str, input_tokens: int, output_tokens: int) -> float:
    """List-price cost in USD, read from PRICES_USD_PER_MTOK and nowhere else."""
    price = PRICES_USD_PER_MTOK[price_key]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def require_text(ticket: str) -> str:
    """Reject an empty ticket with a clear error. Otherwise return the text exactly as given."""
    if not isinstance(ticket, str) or not ticket.strip():
        raise ValueError("Ticket text is empty: pass the customer's message as a non-empty string.")
    return ticket
