"""Builders and fakes shared by the tests. Nothing here touches the network."""
from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx2
from anthropic.types.beta import BetaMessage
from typesafe_sdk import SystemOneResponse

from core import Declined, Draft

DECISION_JSON = json.dumps({"team": "billing", "urgency": "today", "refund_request": True})


def jev_payload(confidence=0.93, odds=None, refund=0.97, usage=(118, 12), model="jev-1.13.0") -> dict:
    """A System One response body shaped like the documented one."""
    odds = odds or {"0": 0.05, "1": 0.15, "2": 0.80}
    rest = 1 - confidence
    return {
        "model": model,
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "confidence": confidence,
                "probabilities": {"billing": confidence, "technical": rest * 0.7, "sales": rest * 0.3},
            },
            "urgency": {
                "type": "score",
                "score": sum(int(level) * p for level, p in odds.items()),
                "confidence": 0.9,
                "legend": {"0": "can wait", "1": "this week", "2": "today"},
                "probabilities": odds,
            },
            "refund_request": {"type": "noul", "noul": refund},
        },
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


def jev_response(**kwargs) -> SystemOneResponse:
    """Decoded by the SDK's own decoder, exactly as a real HTTP response would be."""
    return SystemOneResponse.from_http_response(httpx2.Response(200, json=jev_payload(**kwargs)))


def message_payload(text=DECISION_JSON, stop_reason="end_turn", model="claude-opus-5-5", usage=(311, 57), stop_details=None) -> dict:
    """A Messages API response body. text=None means no content, as in a refusal before any output."""
    content = [] if text is None else [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}]
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


def message(**kwargs) -> BetaMessage:
    """A real SDK message object."""
    return BetaMessage.model_validate(message_payload(**kwargs))


def refusal(category="cyber") -> BetaMessage:
    details = {"type": "refusal", "category": category, "explanation": None}
    return message(text=None, stop_reason="refusal", stop_details=details)


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: records each request and replays canned messages."""

    def __init__(self, *messages: BetaMessage) -> None:
        self.calls: list[dict] = []
        self._messages = list(messages)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> BetaMessage:
        self.calls.append(kwargs)
        return self._messages.pop(0)


def real_anthropic(handler) -> anthropic.Anthropic:
    """The real SDK client, with its HTTP layer replaced. Needs the fake_keys fixture."""
    return anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


class StubJev:
    """A Decider that returns a fixed decision."""

    def __init__(self, decision) -> None:
        self.decision = decision

    def decide(self, ticket: str):
        return self.decision


class SpyWriter:
    """A Writer that records the tickets it was asked about."""

    def __init__(self, error: Declined | None = None) -> None:
        self.calls: list[str] = []
        self.error = error

    def draft_reply(self, ticket: str) -> Draft:
        self.calls.append(ticket)
        if self.error:
            raise self.error
        return Draft("draft")
