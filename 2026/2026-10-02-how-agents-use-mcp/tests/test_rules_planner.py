"""The keyword planner: argument extraction, tool selection, and how failures surface."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from conftest import read_outbox, run_async

import agent


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("Summarise this week's P1 incidents", "P1"),
        ("show p2 incidents", "P2"),
        ("all P4 tickets", "P4"),
        ("what happened this week?", "P1"),  # no priority mentioned: default
        ("anything P9 lately", "P9"),  # unknown level is passed on; the server rejects it
        ("a P15 incident", "P15"),
    ],
)
def test_extract_priority(request_text: str, expected: str) -> None:
    assert agent.extract_priority(request_text) == expected


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("this week's incidents", 7),
        ("incidents from the last 3 days", 3),
        ("incidents in the past 14 days", 14),
        ("incidents today", 1),
        ("incidents in the last 24 hours", 1),
        ("incidents this month", 30),
        ("any incidents", 7),  # default
        ("incidents from the last 120 days", 120),  # out of range on purpose: the server rejects it
    ],
)
def test_extract_days(request_text: str, expected: int) -> None:
    assert agent.extract_days(request_text) == expected


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("email the team", "ops-team"),
        ("email bob.smith+alerts@example.com a summary", "bob.smith+alerts@example.com"),
        ("send it to the on-call list", "ops-team"),
    ],
)
def test_extract_recipient(request_text: str, expected: str) -> None:
    assert agent.extract_recipient(request_text) == expected


def test_compose_email_for_incidents() -> None:
    incidents = [
        {"id": "INC-1", "title": "Boom", "priority": "P1", "service": "svc-a", "status": "open", "opened_at": "x"},
        {"id": "INC-2", "title": "Bang", "priority": "P1", "service": "svc-a", "status": "resolved", "opened_at": "x"},
        {"id": "INC-3", "title": "Pop", "priority": "P1", "service": "svc-b", "status": "resolved", "opened_at": "x"},
    ]
    subject, body = agent.compose_email("P1", 7, incidents, None)
    assert subject == "P1 incident summary - last 7 days (3 incidents)"
    assert "- svc-a: 2" in body and "- svc-b: 1" in body
    assert body.index("svc-a: 2") < body.index("svc-b: 1")  # busiest service first
    assert "- INC-1 [open] svc-a: Boom" in body


def test_compose_email_with_no_incidents_and_singular_wording() -> None:
    subject, body = agent.compose_email("P4", 1, [], None)
    assert subject == "P4 incident summary - last 1 day (0 incidents)"
    assert "No P4 incidents were opened in the last 1 day." in body
    one = {"id": "INC-1", "title": "t", "priority": "P1", "service": "s", "status": "open", "opened_at": "x"}
    assert agent.compose_email("P1", 7, [one], None)[0].endswith("(1 incident)")


def run_rules(request_text: str):
    buffer = io.StringIO()
    run = run_async(agent.run_agent(request_text, "rules", tracer=agent.Tracer(buffer)))
    return run, buffer.getvalue()


def test_a_runbook_question_uses_only_the_docs_server(outbox: Path) -> None:
    run, _trace = run_rules("Find the runbook for checkout-api 5xx errors")
    assert [(call.server, call.name) for call in run.calls] == [("docs", "search_docs")]
    assert "Checkout API: elevated 5xx error rate" in run.answer
    assert not outbox.exists()  # nothing was emailed


def test_a_priority_with_no_matches_is_reported_and_still_emailed(outbox: Path) -> None:
    run, _trace = run_rules("Email the team the P4 incidents from the last 1 day")
    search, send = run.calls
    assert search.arguments == {"priority": "P4", "days": 1}
    assert search.result == []  # zero results: an empty list, no crash
    assert "Found 0 P4 incidents in the last 1 day." in run.answer
    (stored,) = read_outbox(outbox)
    assert "No P4 incidents were opened in the last 1 day." in stored["body"]
    assert send.result["status"] == "sent"


def test_a_server_side_validation_error_stops_the_plan_before_any_email(outbox: Path) -> None:
    run, trace = run_rules("Summarise P1 incidents from the last 120 days and email the team")
    assert [call.name for call in run.calls] == ["search_incidents"]  # send_email never ran
    assert run.calls[0].is_error
    assert run.answer.startswith("search_incidents failed:")
    assert "between 1 and 90" in run.answer
    assert "isError=true" in trace
    assert not outbox.exists()


def test_a_request_the_rules_do_not_cover_fails_cleanly(outbox: Path) -> None:
    with pytest.raises(agent.PlanningError, match="mentions neither"):  # a plain error, not an ExceptionGroup
        run_rules("What is the weather in Lisbon?")
    assert not outbox.exists()


def test_asking_to_email_without_saying_what_to_send_fails_cleanly(outbox: Path) -> None:
    with pytest.raises(agent.PlanningError):
        run_rules("Email the team")
    assert not outbox.exists()
