"""search_incidents: the data, the validation, and how errors look to an MCP client."""

from __future__ import annotations

import json
import re

import pytest
from conftest import open_server, run_async

from servers import incidents_server
from servers.incidents_server import find_incidents

# (priority, days) -> expected number of incidents. The first and fourth rows are the
# numbers the post relies on; the rest pin the dataset so accidental edits are noticed.
EXPECTED_COUNTS = [
    ("P1", 7, 12),
    ("P1", 1, 3),
    ("P1", 8, 13),  # INC-4759 is 190 hours old: just outside 7 days, inside 8
    ("P1", 90, 15),  # the 100-day-old P1 stays outside the 90-day maximum
    ("P2", 7, 5),
    ("P3", 7, 4),
    ("P4", 7, 2),
    ("P4", 1, 0),  # the "zero results" case
]


@pytest.mark.parametrize(("priority", "days", "count"), EXPECTED_COUNTS)
def test_result_counts(priority: str, days: int, count: int) -> None:
    assert len(find_incidents(priority, days)) == count


def test_incidents_are_well_formed_newest_first_and_inside_the_window() -> None:
    incidents = find_incidents("P1", 7)
    assert [i["id"] for i in incidents] == sorted((i["id"] for i in incidents), reverse=True)
    assert "INC-4821" in {i["id"] for i in incidents}
    for incident in incidents:
        assert set(incident) == {"id", "title", "priority", "service", "status", "opened_at"}
        assert re.fullmatch(r"INC-\d{4}", incident["id"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", incident["opened_at"])
        assert incident["opened_at"] >= "2026-09-25T09:00:00Z"  # 7 days before the fixed reference time


def test_the_dataset_is_deterministic() -> None:
    assert find_incidents("P1", 7) == find_incidents("P1", 7)
    assert incidents_server.REFERENCE_NOW.year == 2026  # anchored to a fixed date, not to today


@pytest.mark.parametrize("priority", ["P9", "P0", "p1", " P1", "", "HIGH", "P1; DROP TABLE incidents"])
def test_unknown_priority_is_a_value_error_naming_the_valid_ones(priority: str) -> None:
    with pytest.raises(ValueError, match=r"must be one of P1, P2, P3, P4"):
        find_incidents(priority, 7)


@pytest.mark.parametrize("days", [0, -1, 91, 10_000])
def test_days_outside_1_to_90_is_a_value_error(days: int) -> None:
    with pytest.raises(ValueError, match=r"between 1 and 90"):
        find_incidents("P1", days)


def test_a_very_long_priority_is_not_echoed_in_full() -> None:
    with pytest.raises(ValueError) as caught:
        find_incidents("X" * 5000, 7)
    assert len(str(caught.value)) < 200


# ---- the same checks through a real MCP session -------------------------------------


def test_over_mcp_p1_seven_days_is_one_json_text_block_with_12_incidents() -> None:
    async def call():
        async with open_server("incidents") as session:
            return await session.call_tool("search_incidents", {"priority": "P1", "days": 7})

    result = run_async(call())
    assert not result.is_error
    assert len(result.content) == 1 and result.content[0].type == "text"
    assert len(json.loads(result.content[0].text)) == 12


def test_over_mcp_zero_matches_is_an_empty_list_not_an_error() -> None:
    async def call():
        async with open_server("incidents") as session:
            return await session.call_tool("search_incidents", {"priority": "P4", "days": 1})

    result = run_async(call())
    assert not result.is_error
    assert json.loads(result.content[0].text) == []


def test_over_mcp_bad_input_gives_a_readable_error_and_the_server_keeps_working() -> None:
    bad_calls = [
        {"priority": "P9", "days": 7},
        {"priority": "P1", "days": 0},
        {"priority": "P1", "days": 91},
        {"priority": "P1", "days": "seven"},  # wrong type: rejected by the input schema
        {"priority": "P1"},  # missing argument
        {},
    ]

    async def call_all():
        async with open_server("incidents") as session:
            errors = [await session.call_tool("search_incidents", args) for args in bad_calls]
            still_alive = await session.call_tool("search_incidents", {"priority": "P1", "days": 7})
            return errors, still_alive

    errors, still_alive = run_async(call_all())

    for result in errors:
        assert result.is_error  # reported inside the result, not as a protocol failure
        assert result.content and result.content[0].text.strip()

    assert "P9" in errors[0].content[0].text and "P1, P2, P3, P4" in errors[0].content[0].text
    assert "between 1 and 90" in errors[1].content[0].text and "between 1 and 90" in errors[2].content[0].text
    assert "days" in errors[3].content[0].text
    assert "days" in errors[4].content[0].text

    assert not still_alive.is_error  # no crash: the next valid call succeeds
    assert len(json.loads(still_alive.content[0].text)) == 12
