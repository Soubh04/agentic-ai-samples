"""Edge cases for the ticket text."""
import json

import httpx2
import pytest
from helpers import FakeAnthropic, jev_payload, message
from typesafe_sdk import TypeSafeClient

from claude_backend import ClaudeBackend
from core import require_text
from jev_backend import JevBackend
from mock_backends import MockClaude, MockJev

# HTML entities, emoji, a zero-width joiner, an accent, and text that looks like format placeholders.
MESSY = "Refund &amp; credit &lt;b&gt;now&lt;/b&gt; &#128512; \U0001F600 \U0001F468‍\U0001F4BB café %s {braces}"
PADDED = "  leading and trailing whitespace is kept \n"


def jev_recording(seen: list) -> JevBackend:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=jev_payload())

    return JevBackend(TypeSafeClient(transport=httpx2.MockTransport(handler)))


@pytest.mark.parametrize("ticket", ["", "   ", "\n\t  \n", None, 42])
def test_an_empty_or_non_text_ticket_is_rejected_with_a_clear_error(ticket):
    with pytest.raises(ValueError, match="Ticket text is empty"):
        require_text(ticket)


def test_every_backend_rejects_an_empty_ticket_before_making_any_request(fake_keys):
    requests, fake = [], FakeAnthropic()
    jev, claude = jev_recording(requests), ClaudeBackend(fake)
    methods = [MockJev().decide, MockClaude().decide, MockClaude().draft_reply, jev.decide, claude.decide, claude.draft_reply]
    for method in methods:
        for ticket in ("", "  \n"):
            with pytest.raises(ValueError, match="Ticket text is empty"):
                method(ticket)
    assert requests == [] and fake.calls == []


@pytest.mark.parametrize("ticket", [MESSY, PADDED])
def test_the_text_passes_through_unchanged(ticket, fake_keys):
    assert require_text(ticket) is ticket
    requests, fake = [], FakeAnthropic(message(), message())
    jev_recording(requests).decide(ticket)
    ClaudeBackend(fake).decide(ticket)
    ClaudeBackend(fake).draft_reply(ticket)
    assert requests[0]["state"] == ticket  # not unescaped, stripped, normalised or truncated
    assert [call["messages"][0]["content"] for call in fake.calls] == [ticket, ticket]


def test_the_mock_backends_accept_the_same_messy_text():
    assert MockJev().decide(MESSY).team
    assert MockClaude().decide(MESSY).team
    assert MockClaude().draft_reply(MESSY).text


def test_a_very_long_ticket_is_not_truncated(fake_keys):
    ticket = "Please look at my invoice. " * 20_000  # about 540,000 characters
    requests, fake = [], FakeAnthropic(message())
    jev_recording(requests).decide(ticket)
    ClaudeBackend(fake).decide(ticket)
    assert requests[0]["state"] == ticket
    assert fake.calls[0]["messages"][0]["content"] == ticket
