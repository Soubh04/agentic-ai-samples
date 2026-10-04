"""Support-ticket triage: Jev decides, code checks, the LLM writes.

    uv run python triage.py             sample tickets (a mock backend for any API key that is not set)
    uv run python triage.py --compare   Jev and Claude side by side on every ticket
    uv run python triage.py --live      require both API keys and never fall back to mocks
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import unicodedata
from dataclasses import dataclass

import anthropic
from typesafe_sdk import TypeSafeError

from claude_backend import ClaudeBackend
from core import BackendError, CallStats, Declined, Decider, Decision, Draft, Writer, require_text
from jev_backend import JevBackend
from mock_backends import MockClaude, MockJev
from tickets import SAMPLE_TICKETS

DEFAULT_THRESHOLD = 0.7
TYPESAFE_KEY, ANTHROPIC_KEY = "TYPESAFE_API_KEY", "ANTHROPIC_API_KEY"


# --- The code check -----------------------------------------------------------------------------
def route(decision: Decision, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True: route on the Jev answer alone. False: escalate to the LLM.

    The comparison is `>=`, so a confidence exactly at the threshold routes. A decision with no
    confidence (Claude returns labels only) is never routed this way.
    """
    return decision.team_confidence is not None and decision.team_confidence >= threshold


@dataclass(frozen=True)
class Outcome:
    decision: Decision
    routed: bool
    draft: Draft | None = None  # escalated, and the LLM wrote a draft
    note: str | None = None  # escalated, but the LLM declined to draft


def triage(ticket: str, jev: Decider, writer: Writer, threshold: float = DEFAULT_THRESHOLD) -> Outcome:
    """Jev decides, code checks, the LLM writes."""
    decision = jev.decide(ticket)
    if route(decision, threshold):
        return Outcome(decision, routed=True)
    try:
        return Outcome(decision, routed=False, draft=writer.draft_reply(ticket))
    except Declined as exc:
        return Outcome(decision, routed=False, note=f"{exc}. Hand this ticket to a person.")


# --- Output -------------------------------------------------------------------------------------
def clean(text: str) -> str:
    """Display only: drop control characters, such as terminal escapes, from untrusted text."""
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def fmt_refund(d: Decision) -> str:
    answer = "yes" if d.refund_request else "no"
    return answer if d.refund_probability is None else f"{answer} ({d.refund_probability:.2f})"


def fmt_stats(stats: CallStats | None) -> str:
    """Latency, tokens and cost, shown only when a real call measured them."""
    if stats is None:
        return ""
    tokens = "tokens n/a" if stats.input_tokens is None else f"{stats.input_tokens}/{stats.output_tokens} tok"
    cost = "cost n/a" if stats.cost_usd is None else f"${stats.cost_usd:.6f}"
    return f"  [{stats.latency_ms:.0f} ms, {tokens}, {cost}]"


def banner(jev: Decider, claude: Decider, threshold: float | None) -> str:
    mock = {"Jev": isinstance(jev, MockJev), "Claude": isinstance(claude, MockClaude)}
    lines = ["Backends: " + ", ".join(f"{name} = {'mock (offline)' if m else 'live API'}" for name, m in mock.items())]
    if any(mock.values()):
        lines.append("Mock backends are keyword rules, not models. They make no API calls, so timings, token counts "
                     "and costs are NOT measured and none are shown. Their confidence values are illustrative.")
    if not all(mock.values()):
        lines.append("Live backends make real API calls, billed to your accounts. Latency is wall-clock time as seen by this script.")
    if threshold is not None:
        lines.append(f"Rule: team confidence >= {threshold:.2f} is routed in code. Lower is escalated to Claude for a draft reply.")
    return "\n".join(textwrap.fill(line, 100) for line in lines) + "\n"


def run_triage(tickets: list[str], jev: Decider, writer: Writer, threshold: float) -> None:
    routed = 0
    for n, ticket in enumerate(tickets, 1):
        outcome = triage(ticket, jev, writer, threshold)
        d = outcome.decision
        confidence = "n/a" if d.team_confidence is None else f"{d.team_confidence:.2f}"
        if outcome.routed:
            routed += 1
            print(f"{n}. {'ROUTED':<9} {d.team:<9} urgency={d.urgency:<9} refund={fmt_refund(d):<10}  "
                  f"team confidence {confidence} >= {threshold:.2f}{fmt_stats(d.stats)}")
        else:
            odds = ", ".join(f"{team} {p:.2f}" for team, p in sorted((d.team_probabilities or {}).items(), key=lambda kv: -kv[1]))
            print(f"{n}. {'ESCALATED':<9} team confidence {confidence} < {threshold:.2f} ({odds}){fmt_stats(d.stats)}")
        print(f'   "{textwrap.shorten(clean(ticket), 120, placeholder="…")}"')
        if outcome.draft:
            print(f"   Claude's draft, for a human to review:{fmt_stats(outcome.draft.stats)}")
            print(textwrap.indent(clean(outcome.draft.text), "     "))
        elif outcome.note:
            print(f"   {outcome.note}")
    print(f"\nTickets: {len(tickets)} | routed in code: {routed} | escalated to the LLM: {len(tickets) - routed}")


def measured(stats: CallStats | None) -> tuple[str, str, str]:
    """Table cells for latency, tokens and cost. "-" means nothing was measured."""
    if stats is None:
        return "-", "-", "-"
    return (
        f"{stats.latency_ms:.0f} ms",
        "n/a" if stats.input_tokens is None else f"{stats.input_tokens} / {stats.output_tokens}",
        "n/a" if stats.cost_usd is None else f"${stats.cost_usd:.6f}",
    )


def run_compare(tickets: list[str], jev: Decider, claude: Decider) -> None:
    rows = [("#", "backend", "team", "urgency", "refund", "latency", "tokens in/out", "cost (USD)")]
    for n, ticket in enumerate(tickets, 1):
        for label, backend in (("Jev", jev), ("Claude", claude)):
            try:
                d = backend.decide(ticket)
            except Declined as exc:
                rows.append((str(n), label, f"declined ({exc.category or 'no category'})", "", "", "", "", ""))
                continue
            name = f"{label} ({d.stats.model})" if d.stats else label  # the model the API says served the call
            rows.append((str(n), name, d.team, d.urgency, fmt_refund(d), *measured(d.stats)))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
        if i == 0:
            print("  ".join("-" * width for width in widths))
    cells = [cell for row in rows[1:] for cell in row[5:]]
    print()
    if "-" in cells:
        print('"-" means not measured: a mock backend made no API call.')
    if "n/a" in cells:
        print('"n/a" means the API gave no token counts, or no published price covers the model that served the call.')


# --- Command line -------------------------------------------------------------------------------
def build_backends(offline: bool = False) -> tuple[JevBackend | MockJev, ClaudeBackend | MockClaude]:
    """A real backend for each API key that is set, a mock for each key that is not."""
    jev = JevBackend() if os.environ.get(TYPESAFE_KEY) and not offline else MockJev()
    claude = ClaudeBackend() if os.environ.get(ANTHROPIC_KEY) and not offline else MockClaude()
    return jev, claude


def probability(text: str) -> float:
    """argparse type for --threshold: a number from 0 to 1."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="require both API keys; never fall back to mocks")
    mode.add_argument("--offline", action="store_true", help="use the mock backends even if API keys are set")
    parser.add_argument("--compare", action="store_true", help="run Jev and Claude side by side on every ticket")
    parser.add_argument("--threshold", type=probability, default=DEFAULT_THRESHOLD, metavar="0-1",
                        help=f"team confidence needed to route in code (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--ticket", help="triage this text instead of the sample tickets")
    args = parser.parse_args(argv)

    missing = [var for var in (TYPESAFE_KEY, ANTHROPIC_KEY) if not os.environ.get(var)]
    if args.live and missing:
        parser.error(f"--live needs both API keys in the environment; missing: {', '.join(missing)}")
    try:
        tickets = SAMPLE_TICKETS if args.ticket is None else [require_text(args.ticket)]
    except ValueError as exc:
        parser.error(str(exc))

    jev, claude = build_backends(args.offline)
    print(banner(jev, claude, None if args.compare else args.threshold))
    try:
        if args.compare:
            run_compare(tickets, jev, claude)
        else:
            run_triage(tickets, jev, claude, args.threshold)
    except (BackendError, TypeSafeError, anthropic.AnthropicError) as exc:  # both SDKs retry transient errors themselves
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
