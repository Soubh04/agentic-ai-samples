"""Offline stand-ins for the two backends: keyword rules, not models.

They implement the same two interfaces as the real backends (Decider and Writer in core.py), so the
sample runs with no network and no keys. They never report latency, tokens or cost, and their
confidence values are illustrative: do not read them as Jev or Claude output.
"""
from __future__ import annotations

from core import Decision, Draft, require_text

TEAM_WORDS = {
    "billing": ("charge", "invoice", "refund", "payment", "billed", "credit"),
    "technical": ("error", "crash", "login", "log in", "bug", "outage", "timeout", "500"),
    "sales": ("pricing", "quote", "upgrade", "demo", "enterprise", "plan"),
}
TODAY_WORDS = ("today", "asap", "urgent", "immediately", "blocked")
WEEK_WORDS = ("this week", "by friday", "next few days")
REFUND_WORDS = ("refund", "money back", "credit")


def _read(ticket: str) -> tuple[dict[str, float], str, bool]:
    """Keyword counts -> team probabilities, an urgency label and a refund flag."""
    text = require_text(ticket).lower()
    hits = {team: sum(text.count(word) for word in words) for team, words in TEAM_WORDS.items()}
    total = sum(hits.values())
    odds = {team: (count + 0.5) / (total + 1.5) for team, count in hits.items()}  # sums to 1
    if any(word in text for word in TODAY_WORDS):
        urgency = "today"
    elif any(word in text for word in WEEK_WORDS):
        urgency = "this week"
    else:
        urgency = "can wait"
    return odds, urgency, any(word in text for word in REFUND_WORDS)


class MockJev:
    """Stands in for JevBackend: a label plus a confidence."""

    def decide(self, ticket: str) -> Decision:
        odds, urgency, refund = _read(ticket)
        team = max(odds, key=odds.get)  # a tie goes to the first team listed
        return Decision(team, urgency, refund, team_confidence=odds[team], team_probabilities=odds,
                        refund_probability=0.9 if refund else 0.1)


class MockClaude:
    """Stands in for ClaudeBackend: labels only (no probabilities), and a canned draft."""

    def decide(self, ticket: str) -> Decision:
        odds, urgency, refund = _read(ticket)
        return Decision(max(odds, key=odds.get), urgency, refund)

    def draft_reply(self, ticket: str) -> Draft:
        odds, _, _ = _read(ticket)
        return Draft(
            "[mock draft: no model was called]\n"
            f"Likely owner: {max(odds, key=odds.get)} (keyword guess).\n"
            "Draft reply: Thanks for getting in touch. A member of our team will read your\n"
            "message and reply soon. Could you tell us a little more about what you need?"
        )
