"""The Claude side. Unit tests use a fake client; two tests use the real SDK over a mock transport."""
import json
import os

import httpx2
import pytest
from anthropic.types.beta import BetaMessage
from helpers import FakeAnthropic, message, message_payload, real_anthropic, refusal

from claude_backend import DECIDE_SYSTEM, DECISION_SCHEMA, DEFAULT_MODEL, FALLBACK_BETA, ClaudeBackend
from core import PRICES_USD_PER_MTOK, TEAMS, URGENCY, BackendError, Declined, cost_of

TICKET = "I was charged twice for my invoice."


def decide(*messages):
    """Run one decision through a fake client and return (decision, the request that was sent)."""
    client = FakeAnthropic(*messages)
    return ClaudeBackend(client).decide(TICKET), client.calls[0]


# --- The request ---------------------------------------------------------------------------------
def test_request_has_no_temperature_and_no_forced_tool_choice():
    _, request = decide(message())
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "thinking", "tools"):
        assert forbidden not in request


def test_default_model_is_opus_5_5():
    _, request = decide(message())
    assert DEFAULT_MODEL == "claude-opus-5-5"
    assert request["model"] == "claude-opus-5-5"


def test_anthropic_model_overrides_the_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    _, request = decide(message(model="claude-sonnet-5-5"))
    assert request["model"] == "claude-sonnet-5-5"


def test_an_empty_anthropic_model_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    assert decide(message())[1]["model"] == DEFAULT_MODEL


def test_an_explicit_model_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "from-environment")
    client = FakeAnthropic(message())
    ClaudeBackend(client, model="from-argument").decide(TICKET)
    assert client.calls[0]["model"] == "from-argument"


def test_decision_uses_structured_outputs_with_the_shared_rubric():
    _, request = decide(message())
    output_format = request["output_config"]["format"]
    assert output_format == {"type": "json_schema", "schema": DECISION_SCHEMA}
    assert DECISION_SCHEMA["properties"]["team"]["enum"] == list(TEAMS)
    assert DECISION_SCHEMA["properties"]["urgency"]["enum"] == list(URGENCY)
    assert DECISION_SCHEMA["additionalProperties"] is False
    assert request["output_config"]["effort"] == "low"
    for name in (*TEAMS, *URGENCY):
        assert name in DECIDE_SYSTEM


def test_the_ticket_is_the_user_message_exactly():
    _, request = decide(message())
    assert request["messages"] == [{"role": "user", "content": TICKET}]


def test_the_server_side_fallback_is_opted_in():
    _, request = decide(message())
    assert request["fallbacks"] == "default"
    assert request["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert request["max_tokens"] >= 4096  # thinking counts toward max_tokens on this model


def test_the_real_sdk_puts_the_expected_request_on_the_wire(fake_keys):
    """A fake client accepts any keyword. The real SDK rejects unknown ones, so this catches typos."""
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(path=request.url.path, query=request.url.query.decode(), beta=request.headers["anthropic-beta"],
                    key=request.headers["x-api-key"], body=json.loads(request.content))
        return httpx2.Response(200, json=message_payload())

    decision = ClaudeBackend(real_anthropic(handler)).decide(TICKET)
    assert (seen["path"], seen["query"]) == ("/v1/messages", "beta=true")
    assert seen["beta"] == FALLBACK_BETA
    assert seen["key"] == os.environ["ANTHROPIC_API_KEY"]  # read from the environment
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["messages"] == [{"role": "user", "content": TICKET}]
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "thinking", "tools"):
        assert forbidden not in body
    assert decision.team == "billing"


# --- The response --------------------------------------------------------------------------------
def test_decision_is_read_from_the_json_text_block_and_thinking_blocks_are_skipped():
    decision, _ = decide(message())  # the canned message starts with an empty thinking block
    assert (decision.team, decision.urgency, decision.refund_request) == ("billing", "today", True)
    assert decision.team_confidence is None  # an LLM returns a label, not a probability
    assert decision.refund_probability is None


def test_a_refusal_is_reported_with_its_category_and_not_read_as_content():
    with pytest.raises(Declined) as caught:
        ClaudeBackend(FakeAnthropic(refusal("cyber"))).decide(TICKET)
    assert caught.value.category == "cyber"
    assert "cyber" in str(caught.value)


def test_a_refusal_without_details_still_does_not_crash():
    bare = message(text=None, stop_reason="refusal")
    with pytest.raises(Declined) as caught:
        ClaudeBackend(FakeAnthropic(bare)).decide(TICKET)
    assert caught.value.category is None


def test_a_refusal_while_drafting_is_also_reported():
    with pytest.raises(Declined):
        ClaudeBackend(FakeAnthropic(refusal("bio"))).draft_reply(TICKET)


def test_a_truncated_reply_is_a_clear_error_not_a_json_traceback():
    truncated = message(text='{"team": "bill', stop_reason="max_tokens")
    with pytest.raises(BackendError, match="max_tokens"):
        ClaudeBackend(FakeAnthropic(truncated)).decide(TICKET)


def test_a_reply_served_by_a_fallback_model_is_read_and_left_unpriced():
    payload = message_payload(model="claude-opus-4-8")
    payload["content"].insert(0, {"type": "fallback", "from": {"model": "claude-opus-5-5"},
                                  "to": {"model": "claude-opus-4-8"}, "trigger": {"type": "refusal", "category": "cyber"}})
    decision, _ = decide(BetaMessage.model_validate(payload))
    assert (decision.team, decision.stats.model, decision.stats.cost_usd) == ("billing", "claude-opus-4-8", None)


def test_a_reply_split_across_text_blocks_is_joined():
    payload = message_payload()
    payload["content"] = [{"type": "text", "text": '{"team": "billing", "urgency": '},
                          {"type": "text", "text": '"today", "refund_request": true}'}]
    decision, _ = decide(BetaMessage.model_validate(payload))
    assert (decision.team, decision.urgency, decision.refund_request) == ("billing", "today", True)


def test_a_reply_without_a_text_block_is_a_clear_error():
    with pytest.raises(BackendError, match="no text"):
        ClaudeBackend(FakeAnthropic(message(text=None))).decide(TICKET)


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"team": "legal", "urgency": "today", "refund_request": True}),
        json.dumps({"team": "billing", "urgency": "whenever", "refund_request": True}),
        json.dumps({"team": "billing", "urgency": "today", "refund_request": "yes"}),
        json.dumps({"team": "billing"}),
        json.dumps(["billing"]),
    ],
)
def test_an_unusable_json_reply_is_a_clear_error(text):
    with pytest.raises(BackendError):
        ClaudeBackend(FakeAnthropic(message(text=text))).decide(TICKET)


# --- Stats and cost ------------------------------------------------------------------------------
def test_stats_come_from_the_response_and_the_price_constants():
    decision, _ = decide(message(usage=(311, 57)))
    stats = decision.stats
    assert (stats.model, stats.input_tokens, stats.output_tokens) == ("claude-opus-5-5", 311, 57)
    assert stats.latency_ms >= 0
    assert stats.cost_usd == cost_of("claude-opus-5-5", 311, 57)
    assert stats.cost_usd == pytest.approx((311 * 4.00 + 57 * 20.00) / 1_000_000)


def test_no_cost_is_computed_when_another_model_served_the_call():
    # For example a server-side fallback: the sample only holds a published price for Opus 5.5.
    decision, _ = decide(message(model="claude-opus-4-8"))
    assert decision.stats.model == "claude-opus-4-8"
    assert decision.stats.cost_usd is None


def test_the_backend_prices_calls_from_the_constants_block(monkeypatch):
    monkeypatch.setitem(PRICES_USD_PER_MTOK, "claude-opus-5-5", {"input": 1.0, "output": 2.0})
    decision, _ = decide(message(usage=(1_000_000, 1_000_000)))
    assert decision.stats.cost_usd == pytest.approx(3.0)


# --- Drafting ------------------------------------------------------------------------------------
def test_draft_reply_returns_the_text_with_stats_and_uses_medium_effort():
    client = FakeAnthropic(message(text="Likely owner: billing.\nDraft reply: Hello."))
    draft = ClaudeBackend(client).draft_reply(TICKET)
    request = client.calls[0]
    assert draft.text.startswith("Likely owner: billing.")
    assert draft.stats.input_tokens == 311
    assert request["output_config"] == {"effort": "medium"}  # no JSON schema for free text
    assert request["messages"] == [{"role": "user", "content": TICKET}]
    for forbidden in ("temperature", "tool_choice", "thinking"):
        assert forbidden not in request
