"""search_docs: the mock runbook library."""

from __future__ import annotations

import json

import pytest
from conftest import open_server, run_async

from servers.docs_server import MAX_RESULTS, find_docs


def test_finds_the_runbook_for_a_service_and_symptom() -> None:
    results = find_docs("Find the runbook for checkout-api 5xx errors")
    assert results[0]["title"] == "Checkout API: elevated 5xx error rate"
    assert results[0]["url"].startswith("https://docs.example.com/runbooks/")


def test_a_query_that_matches_nothing_is_an_empty_list() -> None:
    assert find_docs("quantum entanglement") == []


def test_results_are_capped_and_best_match_first() -> None:
    results = find_docs("incident service database queue cache search login payments")
    assert 0 < len(results) <= MAX_RESULTS


@pytest.mark.parametrize("query", ["", "   ", "!!! ???"])
def test_a_query_without_words_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="at least one word"):
        find_docs(query)


def test_over_mcp_the_tool_returns_a_json_array_and_reports_bad_input() -> None:
    async def call():
        async with open_server("docs") as session:
            ok = await session.call_tool("search_docs", {"query": "orders db failover"})
            nothing = await session.call_tool("search_docs", {"query": "zzzz"})
            blank = await session.call_tool("search_docs", {"query": " "})
            return ok, nothing, blank

    ok, nothing, blank = run_async(call())
    assert json.loads(ok.content[0].text)[0]["title"] == "Orders database: replica failover"
    assert json.loads(nothing.content[0].text) == [] and not nothing.is_error
    assert blank.is_error and "at least one word" in blank.content[0].text
