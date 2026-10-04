"""Cost math, the price block, and the offline mock backends."""
import inspect
import re
from pathlib import Path

import pytest
from helpers import jev_response

from claude_backend import ClaudeBackend
from core import PRICES_CHECKED, PRICES_USD_PER_MTOK, TEAMS, URGENCY, Decider, Writer, cost_of
from jev_backend import JevBackend, to_decision
from mock_backends import TEAM_WORDS, MockClaude, MockJev
from tickets import SAMPLE_TICKETS

ROOT = Path(__file__).resolve().parent.parent


# --- Cost ----------------------------------------------------------------------------------------
def test_cost_math_matches_the_published_rates():
    # Claude Opus 5.5: $4 per million input tokens, $20 per million output tokens.
    assert cost_of("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.00)
    assert cost_of("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.00)
    assert cost_of("claude-opus-5-5", 311, 57) == pytest.approx(0.002384)
    # Jev: $0.042 per million input tokens, and output tokens are free.
    assert cost_of("typesafe-jev", 1_000_000, 0) == pytest.approx(0.042)
    assert cost_of("typesafe-jev", 0, 5_000_000) == 0
    assert cost_of("typesafe-jev", 118, 12) == pytest.approx(118 * 0.042 / 1_000_000)


def test_cost_reads_only_the_constants_block(monkeypatch):
    monkeypatch.setitem(PRICES_USD_PER_MTOK, "typesafe-jev", {"input": 10.0, "output": 100.0})
    assert cost_of("typesafe-jev", 1_000_000, 1_000_000) == pytest.approx(110.0)
    # The Jev adapter prices its calls from the same block.
    stats = to_decision(jev_response(usage=(1_000_000, 1_000_000)), 1.0).stats
    assert stats.cost_usd == pytest.approx(110.0)


def test_an_unknown_price_key_is_an_error_not_a_zero():
    with pytest.raises(KeyError):
        cost_of("no-such-model", 1, 1)


def test_the_price_block_carries_its_check_date():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", PRICES_CHECKED)
    assert set(PRICES_USD_PER_MTOK) == {"typesafe-jev", "claude-opus-5-5"}


def test_price_literals_appear_in_the_constants_block_only():
    for path in ROOT.glob("*.py"):
        if path.name != "core.py":
            text = path.read_text(encoding="utf-8")
            assert "0.042" not in text and "20.00" not in text, f"price literal found in {path.name}"


# --- Mocks ---------------------------------------------------------------------------------------
def test_mocks_and_real_backends_implement_the_same_two_interfaces():
    assert isinstance(MockJev(), Decider)
    assert isinstance(MockClaude(), Decider) and isinstance(MockClaude(), Writer)
    assert isinstance(JevBackend(client=object()), Decider)
    assert isinstance(ClaudeBackend(client=object()), Decider) and isinstance(ClaudeBackend(client=object()), Writer)


@pytest.mark.parametrize(("real", "mock"), [(JevBackend, MockJev), (ClaudeBackend, MockClaude)])
def test_mocks_have_the_same_method_signatures_as_the_real_backends(real, mock):
    for name in ("decide", "draft_reply"):
        if hasattr(real, name):
            assert inspect.signature(getattr(mock, name)) == inspect.signature(getattr(real, name))


def test_mock_backends_are_deterministic_and_measure_nothing():
    for ticket in SAMPLE_TICKETS:
        first, second = MockJev().decide(ticket), MockJev().decide(ticket)
        assert first == second
        assert first.stats is None
        assert MockClaude().decide(ticket).stats is None
        assert MockClaude().draft_reply(ticket) == MockClaude().draft_reply(ticket)
        assert MockClaude().draft_reply(ticket).stats is None


def test_mock_jev_returns_a_confidence_and_mock_claude_returns_labels_only():
    jev, claude = MockJev().decide(SAMPLE_TICKETS[0]), MockClaude().decide(SAMPLE_TICKETS[0])
    assert 0 < jev.team_confidence <= 1
    assert sum(jev.team_probabilities.values()) == pytest.approx(1.0)
    assert (claude.team_confidence, claude.team_probabilities, claude.refund_probability) == (None, None, None)
    assert (jev.team, jev.urgency, jev.refund_request) == (claude.team, claude.urgency, claude.refund_request)


def test_mock_vocabulary_matches_the_rubric():
    assert set(TEAM_WORDS) == set(TEAMS)
    for ticket in SAMPLE_TICKETS:
        decision = MockJev().decide(ticket)
        assert decision.team in TEAMS
        assert decision.urgency in URGENCY
