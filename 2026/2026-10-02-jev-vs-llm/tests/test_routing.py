"""The code check: route in code when Jev is confident, escalate to the LLM when it is not."""
import pytest
from helpers import SpyWriter, StubJev

from core import Declined, Decision
from triage import DEFAULT_THRESHOLD, route, triage


def decision(confidence):
    return Decision(team="billing", urgency="today", refund_request=False, team_confidence=confidence)


def test_default_threshold_is_0_7():
    assert DEFAULT_THRESHOLD == 0.7


def test_high_confidence_is_routed_in_code_and_the_llm_is_not_called():
    writer = SpyWriter()
    outcome = triage("ticket", StubJev(decision(0.93)), writer)
    assert outcome.routed
    assert outcome.draft is None
    assert writer.calls == []


def test_low_confidence_is_escalated_to_the_llm_with_the_original_ticket():
    writer = SpyWriter()
    outcome = triage("ticket text", StubJev(decision(0.41)), writer)
    assert not outcome.routed
    assert outcome.draft.text == "draft"
    assert writer.calls == ["ticket text"]


def test_confidence_exactly_at_the_threshold_routes():
    # Documented rule: "at or above" the threshold routes. Regression guard: changing >= to > fails here.
    assert route(decision(0.7)) is True
    assert route(decision(0.7000001)) is True
    assert route(decision(0.6999999)) is False


@pytest.mark.parametrize(
    ("threshold", "confidence", "routed"),
    [(0.9, 0.85, False), (0.9, 0.9, True), (0.0, 0.0, True), (1.0, 0.99, False), (1.0, 1.0, True)],
)
def test_threshold_is_configurable(threshold, confidence, routed):
    assert route(decision(confidence), threshold) is routed
    assert triage("t", StubJev(decision(confidence)), SpyWriter(), threshold).routed is routed


@pytest.mark.parametrize("confidence", [None, float("nan")])
def test_a_missing_or_nan_confidence_is_never_routed(confidence):
    assert route(decision(confidence)) is False


def test_a_refusal_while_drafting_does_not_crash_the_pipeline():
    writer = SpyWriter(error=Declined("cyber"))
    outcome = triage("t", StubJev(decision(0.2)), writer)
    assert not outcome.routed
    assert outcome.draft is None
    assert "cyber" in outcome.note
