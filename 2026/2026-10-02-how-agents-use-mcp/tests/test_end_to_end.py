"""The whole flow from the post, in rules mode, over real stdio servers.

One agent run is shared by every test in this module (starting three servers takes a
couple of seconds), and each test asserts one fact about it.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import read_outbox, run_async

import agent


@dataclass
class SampleRun:
    run: agent.AgentRun
    trace: str
    outbox: Path


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> SampleRun:
    outbox = tmp_path_factory.mktemp("e2e") / "outbox.jsonl"
    buffer = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EMAIL_OUTBOX_PATH", str(outbox))
        run = run_async(agent.run_agent(agent.SAMPLE_REQUEST, "rules", tracer=agent.Tracer(buffer)))
    return SampleRun(run, buffer.getvalue(), outbox)


def test_tools_are_discovered_on_all_three_servers(sample: SampleRun) -> None:
    assert sample.run.discovered == {
        "incidents": ["search_incidents"],
        "email": ["send_email"],
        "docs": ["search_docs"],
    }


def test_p1_incidents_of_the_last_7_days_returns_exactly_12(sample: SampleRun) -> None:
    search = sample.run.calls[0]
    assert (search.server, search.name) == ("incidents", "search_incidents")
    assert search.arguments == {"priority": "P1", "days": 7}
    assert not search.is_error
    assert len(search.result) == 12
    assert {incident["priority"] for incident in search.result} == {"P1"}
    assert len({incident["id"] for incident in search.result}) == 12  # no duplicates


def test_email_reports_sent_and_the_message_is_in_the_outbox(sample: SampleRun) -> None:
    send = sample.run.calls[1]
    assert (send.server, send.name) == ("email", "send_email")
    assert not send.is_error
    assert send.result["status"] == "sent"
    assert re.fullmatch(r"msg-[0-9a-f]{12}", send.result["message_id"])

    (stored,) = read_outbox(sample.outbox)
    assert stored["message_id"] == send.result["message_id"]
    assert stored["to"] == "ops-team"
    assert "P1" in stored["subject"] and "12" in stored["subject"]
    assert "INC-4821" in stored["body"]  # the summary really contains incident data


def test_the_docs_tool_is_discovered_but_never_called(sample: SampleRun) -> None:
    assert "search_docs" in sample.run.discovered["docs"]
    assert "search_docs" not in [call.name for call in sample.run.calls]
    assert "docs" not in [call.server for call in sample.run.calls]


def test_calls_happen_in_order_search_then_send(sample: SampleRun) -> None:
    assert [call.name for call in sample.run.calls] == ["search_incidents", "send_email"]


def test_the_agent_answers_the_user(sample: SampleRun) -> None:
    answer = sample.run.answer
    assert "12 P1 incidents" in answer
    assert "ops-team" in answer
    assert "status: sent" in answer


def test_trace_follows_the_eight_steps_of_the_post(sample: SampleRun) -> None:
    steps = re.findall(r"^Step (\d+)\s+(.+?)\s{2,}(\S+)", sample.trace, flags=re.MULTILINE)
    assert [(int(number), kind) for number, _direction, kind in steps] == [
        (1, "request"),
        (2, "tools/list"),
        (3, "result"),
        (4, "tools/call"),
        (5, "result"),
        (6, "tools/call"),
        (7, "result"),
        (8, "answer"),
    ]
    assert "search_incidents" in sample.trace and "send_email" in sample.trace
    assert "initialize ok" in sample.trace  # the handshake is shown before tools/list
