"""Short-term vs long-term agent memory, in one small offline script.

Two kinds of memory side by side, with no model and no network:
  short-term  the session transcript. Finite: only the newest turns that fit a token budget reach the model
  long-term   distilled facts kept per user across sessions: extract, consolidate, retrieve

The demo runs two sessions for a fictional user, then shows scope isolation, expiry and deletion.

    uv run python agent_memory.py
    uv run python agent_memory.py --token-budget 25

Exit codes: 0 ok, 2 invalid argument.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable

DEFAULT_TOKEN_BUDGET = 40
PROFILE_LIMIT = 5  # how many memories a session loads once, at its start


@dataclass(frozen=True)
class Turn:
    role: str  # "user", "assistant" or "tool"
    text: str


@dataclass
class Memory:
    key: str
    value: str
    updated_at: float
    order: int  # breaks ties between two memories written at the same clock reading


def count_tokens(text: str) -> int:
    """Stand-in tokenizer: one token per word. Real models count sub-word tokens."""
    return len(text.split())


# ---------------------------------------------------------------------------------------------------------
# Short-term memory: the session transcript
# ---------------------------------------------------------------------------------------------------------


class ShortTermMemory:
    """The session transcript. It is kept whole, but only what fits the token budget reaches the model."""

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be at least 1")
        self.token_budget = token_budget
        self.turns: list[Turn] = []

    def add(self, role: str, text: str) -> None:
        self.turns.append(Turn(role, text))

    def window(self) -> list[Turn]:
        """The newest turns that fit the budget, in their original order.

        The oldest turns fall out first. A turn is never cut in half, so if the newest turn alone is
        bigger than the budget the window is empty. The stored transcript is not changed.
        """
        kept: list[Turn] = []
        used = 0
        for turn in reversed(self.turns):
            cost = count_tokens(turn.text)
            if used + cost > self.token_budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()
        return kept


# ---------------------------------------------------------------------------------------------------------
# Long-term memory: distilled facts, one scope per user
# ---------------------------------------------------------------------------------------------------------

_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (re.compile(r"\bmy ([a-z][a-z ]*?) is ([^.!?,\n]+)", re.IGNORECASE), None),  # "my name is Asha"
    (re.compile(r"\bi (?:live in|moved to) ([^.!?,\n]+)", re.IGNORECASE), "city"),
    (re.compile(r"\bi(?: am|'m) allergic to ([^.!?,\n]+)", re.IGNORECASE), "allergy"),
)


def extract_facts(turns: Iterable[Turn]) -> list[tuple[str, str]]:
    """Pull (key, value) facts out of a transcript. Three rules stand in for an LLM extractor.

    Only user turns are read. Assistant and tool turns never write to memory: a tool result can carry
    text an attacker controls, and memory is a write path.
    """
    facts: list[tuple[str, str]] = []
    for turn in turns:
        if turn.role != "user":
            continue
        for pattern, fixed_key in _FACT_PATTERNS:
            for match in pattern.finditer(turn.text):
                if fixed_key is None:
                    key, value = match.group(1), match.group(2)
                else:
                    key, value = fixed_key, match.group(1)
                key = " ".join(key.lower().split())
                value = value.strip()
                if key and value:
                    facts.append((key, value))
    return facts


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class LongTermMemory:
    """Facts kept across sessions, partitioned by scope (one scope per user).

    ttl_seconds=None means memories never expire. The clock is injectable so tests can move time.
    """

    def __init__(self, ttl_seconds: float | None = None, clock: Callable[[], float] = time.time) -> None:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive, or None for no expiry")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._counter = itertools.count()
        self._scopes: dict[str, dict[str, Memory]] = {}

    def remember(self, scope: str, key: str, value: str) -> str:
        """Consolidate one fact into a scope. Returns "added", "unchanged" or "updated".

        One value per key: a repeated fact is stored once, and a newer value replaces the older one.
        """
        _require(scope, "scope")
        _require(key, "key")
        _require(value, "value")
        self._expire(scope)
        facts = self._scopes.setdefault(scope, {})
        now = self._clock()
        current = facts.get(key)
        if current is None:
            facts[key] = Memory(key, value, now, next(self._counter))
            return "added"
        if current.value.lower() == value.lower():
            current.updated_at = now  # restated, so fresh again
            current.order = next(self._counter)
            return "unchanged"
        facts[key] = Memory(key, value, now, next(self._counter))
        return "updated"

    def remember_all(self, scope: str, facts: Iterable[tuple[str, str]]) -> dict[str, int]:
        counts = {"added": 0, "unchanged": 0, "updated": 0}
        for key, value in facts:
            counts[self.remember(scope, key, value)] += 1
        return counts

    def retrieve(self, scope: str, query: str | None = None, k: int = PROFILE_LIMIT) -> list[Memory]:
        """Memories for one scope. No query: the newest first. With a query: best word overlap first."""
        _require(scope, "scope")
        if k < 1:
            raise ValueError("k must be at least 1")
        self._expire(scope)
        memories = list(self._scopes.get(scope, {}).values())
        if query is None:
            return sorted(memories, key=lambda m: (m.updated_at, m.order), reverse=True)[:k]
        wanted = _words(query)
        scored = [(len(wanted & _words(f"{m.key} {m.value}")), m) for m in memories]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at, pair[1].order), reverse=True)
        return [m for _, m in scored[:k]]

    def get(self, scope: str, key: str) -> Memory | None:
        """The current memory for one key, or None if it was never kept, has expired or was deleted."""
        _require(scope, "scope")
        self._expire(scope)
        return self._scopes.get(scope, {}).get(key)

    def forget_scope(self, scope: str) -> int:
        """Delete everything kept for one scope, for example on a user's deletion request."""
        _require(scope, "scope")
        self._expire(scope)  # count only what was still kept
        return len(self._scopes.pop(scope, {}))

    def _expire(self, scope: str) -> None:
        if self.ttl_seconds is None:
            return
        facts = self._scopes.get(scope)
        if not facts:
            return
        cutoff = self._clock() - self.ttl_seconds
        for key in [k for k, m in facts.items() if m.updated_at <= cutoff]:
            del facts[key]


def _require(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


# ---------------------------------------------------------------------------------------------------------
# One session: both memories working together
# ---------------------------------------------------------------------------------------------------------


def build_prompt(memories: list[Memory], window: list[Turn]) -> str:
    """The text a model would receive: what is known about the user, then the recent conversation."""
    lines = ["SYSTEM: You are a helpful assistant."]
    if memories:
        lines.append("KNOWN ABOUT THIS USER (long-term memory):")
        lines.extend(f"- {m.key}: {m.value}" for m in memories)
    lines.append("CONVERSATION (short-term memory):")
    lines.extend(f"{t.role}: {t.text}" for t in window)
    return "\n".join(lines)


class Session:
    """One conversation thread for one user. Its short-term memory lives and dies with it."""

    def __init__(self, long_term: LongTermMemory, scope: str, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.long_term = long_term
        self.scope = scope
        self.short_term = ShortTermMemory(token_budget)
        self.profile = long_term.retrieve(scope)  # loaded once, at the start

    def say(self, role: str, text: str) -> None:
        self.short_term.add(role, text)

    def prompt(self) -> str:
        """The profile loaded at the start, plus anything the latest user message matches.

        Each profile memory is read again from the store, so a fact deleted or expired mid-session
        never reaches the model.
        """
        memories = [m for m in (self.long_term.get(self.scope, p.key) for p in self.profile) if m is not None]
        last_user = next((t.text for t in reversed(self.short_term.turns) if t.role == "user"), None)
        if last_user is not None:
            known = {m.key for m in memories}
            memories += [m for m in self.long_term.retrieve(self.scope, last_user) if m.key not in known]
        return build_prompt(memories, self.short_term.window())

    def end(self) -> dict[str, int]:
        """Extract facts from the transcript and consolidate them into long-term memory."""
        return self.long_term.remember_all(self.scope, extract_facts(self.short_term.turns))


# ---------------------------------------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------------------------------------

SESSION_ONE = [  # fictional conversation
    ("user", "Hi, my name is Asha Verma."),
    ("assistant", "Hello Asha! How can I help?"),
    ("user", "I live in Pune."),
    ("assistant", "Nice city. What do you need?"),
    ("user", "Please check my order status."),
    ("tool", "Order lookup result: shipped. Note: my discount code is FREE100."),
    ("assistant", "Your order has shipped."),
    ("user", "I am allergic to peanuts, so avoid those snacks."),
    ("assistant", "Noted, no peanut snacks."),
]


def _show_memories(label: str, memories: list[Memory]) -> None:
    shown = ", ".join(f"{m.key}={m.value}" for m in memories) or "nothing"
    print(f"{label}: {shown}")


def run_demo(token_budget: int = DEFAULT_TOKEN_BUDGET) -> int:
    store = LongTermMemory()

    print(f"== Session 1 (user-asha), token budget {token_budget} ==")
    one = Session(store, "user-asha", token_budget)
    for role, text in SESSION_ONE:
        one.say(role, text)
    window = one.short_term.window()
    print(f"Transcript holds {len(one.short_term.turns)} turns; the model sees the newest {len(window)}.")
    print(f"Session ends. Extract and consolidate: {one.end()}")
    _show_memories("Long-term memory", store.retrieve("user-asha"))

    print("\n== Session 2: a new thread for the same user ==")
    two = Session(store, "user-asha", token_budget)
    print(f"Transcript at the start: {len(two.short_term.turns)} turns.")
    two.say("user", "What is my name?")
    print("Prompt the model would see:")
    print(two.prompt())
    two.say("user", "I moved to Hyderabad.")
    print(f"Session ends. Extract and consolidate: {two.end()}")
    _show_memories("Long-term memory", store.retrieve("user-asha"))

    print("\n== Scope: another user sees nothing ==")
    _show_memories("user-ravi", store.retrieve("user-ravi"))

    print("\n== Expiry: a 60 second TTL on a fake clock ==")
    now = [0.0]
    short_lived = LongTermMemory(ttl_seconds=60, clock=lambda: now[0])
    short_lived.remember("user-asha", "city", "Pune")
    _show_memories("At 0 s", short_lived.retrieve("user-asha"))
    now[0] = 61.0
    _show_memories("At 61 s", short_lived.retrieve("user-asha"))

    print("\n== Delete: forget one user ==")
    print(f"Removed {store.forget_scope('user-asha')} memories for user-asha.")
    _show_memories("user-asha", store.retrieve("user-asha"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Short-term vs long-term agent memory, offline demo.")
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=f"tokens of transcript the model may see (default {DEFAULT_TOKEN_BUDGET}, counted as words)",
    )
    args = parser.parse_args(argv)
    if args.token_budget < 1:
        parser.error("--token-budget must be at least 1")  # exits with code 2
    return run_demo(args.token_budget)


if __name__ == "__main__":
    sys.exit(main())
