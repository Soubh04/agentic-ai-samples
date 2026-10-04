"""Jev (TypeSafe System One): one call, three typed questions, answers with probabilities."""
from __future__ import annotations

import time

from typesafe_sdk import Choice, Noul, Score, SystemOneResponse, TypeSafeClient

from core import REFUND, TEAMS, URGENCY, BackendError, CallStats, Decision, cost_of, require_text

# Built with the SDK's own classes; no network is needed to construct them.
# Choice: one label from a fixed set. Score: ordered levels (a criterion's position is its level).
# Noul: how likely a statement is to hold, from 0 to 1.
QUESTIONS = {
    "team": Choice(instructions="Which team should handle this support ticket?", criteria=dict(TEAMS)),
    "urgency": Score(instructions="How soon does a human need to act on this ticket?", criteria=list(URGENCY.values())),
    "refund_request": Noul(instructions=REFUND),
}
REFUND_CUTOFF = 0.5  # a Noul answer is a probability; this turns it into the yes/no that is shown


def to_decision(response: SystemOneResponse, latency_ms: float) -> Decision:
    """The adapter: SDK response in, Decision out.

    `.choices`, `.scores` and `.nouls` are typed views of `.answers`. A Choice answer has `choice`,
    `confidence` and `probabilities`. A Score answer has `score` (an expected value that can fall
    between levels), `confidence` and `probabilities`. A Noul answer has only `noul`.
    """
    try:
        team = response.choices["team"]
        urgency = response.scores["urgency"]
        refund = response.nouls["refund_request"]
        odds = urgency.probabilities  # {level: probability}
        # The most likely level, not the rounded expected score: a 50/50 split between levels 0 and 2
        # has an expected score of 1, a level the model does not favour. A tie goes to the more urgent.
        level = max(odds, key=lambda lvl: (odds[lvl], lvl))
        urgency_label = list(URGENCY)[level]
    except (KeyError, IndexError, ValueError) as exc:
        raise BackendError(f"Unexpected Jev response: problem with {exc!r}") from exc
    usage = response.usage
    known = usage.input_tokens is not None and usage.output_tokens is not None
    stats = CallStats(
        model=response.model,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_of("typesafe-jev", usage.input_tokens, usage.output_tokens) if known else None,
    )
    return Decision(
        team=team.choice,
        urgency=urgency_label,
        refund_request=refund.noul >= REFUND_CUTOFF,
        team_confidence=team.confidence,
        team_probabilities=dict(team.probabilities),
        refund_probability=refund.noul,
        stats=stats,
    )


class JevBackend:
    """The real TypeSafe API. The SDK reads TYPESAFE_API_KEY from the environment."""

    def __init__(self, client: TypeSafeClient | None = None) -> None:
        self.client = client or TypeSafeClient()

    def decide(self, ticket: str) -> Decision:
        ticket = require_text(ticket)
        started = time.perf_counter()
        response = self.client.system_one(state=ticket, questions=QUESTIONS)
        return to_decision(response, (time.perf_counter() - started) * 1000)
