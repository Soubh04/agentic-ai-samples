"""The command line, end to end, with no network."""
import re
from pathlib import Path

import anthropic
import httpx2
import pytest
from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAuthenticationError, TypeSafeRateLimitError

import triage
from core import CallStats, Declined, Decision, Draft
from mock_backends import MockClaude, MockJev
from tickets import SAMPLE_TICKETS

ROOT = Path(__file__).resolve().parent.parent
SOME_URL = httpx2.Request("POST", "https://example.invalid/v1")  # .invalid never resolves


def run(capsys, *args):
    code = triage.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class LiveJev:
    """Stands in for JevBackend so key detection can be tested without the network."""

    def decide(self, ticket):
        return Decision("billing", "today", True, 0.9, {"billing": 0.9, "technical": 0.06, "sales": 0.04}, 0.95,
                        CallStats("jev-test", 142.0, 118, 12, 0.000005))


class LiveClaude:
    def decide(self, ticket):
        return Decision("billing", "today", True, stats=CallStats("claude-test", 1800.0, 400, 90, 0.003))

    def draft_reply(self, ticket):
        return Draft("Draft text", CallStats("claude-test", 2500.0, 450, 300, 0.008))


# --- Offline by default ----------------------------------------------------------------------------
def test_the_default_run_is_offline_and_routes_four_of_six(capsys):
    code, out, err = run(capsys)
    assert (code, err) == (0, "")
    assert "Backends: Jev = mock (offline), Claude = mock (offline)" in out
    assert "Tickets: 6 | routed in code: 4 | escalated to the LLM: 2" in out
    assert out.count("ROUTED") == 4 and out.count("ESCALATED") == 2


def test_the_offline_run_is_deterministic(capsys):
    assert run(capsys)[1] == run(capsys)[1]


@pytest.mark.parametrize("args", [[], ["--compare"]])
def test_mock_mode_says_nothing_was_measured_and_prints_no_timings_tokens_or_costs(capsys, args):
    _, out, _ = run(capsys, *args)
    assert "NOT measured" in out
    assert not re.search(r"\d\s*ms\b", out)  # no latency
    assert not re.search(r"\$\s*\d", out)  # no cost
    assert not re.search(r"\d+ ?/ ?\d+\s*tok|\d+ / \d+", out)  # no token counts


def test_compare_offline_prints_one_row_per_ticket_and_backend(capsys):
    code, out, _ = run(capsys, "--compare")
    header = next(line for line in out.splitlines() if line.startswith("#"))
    assert code == 0
    for column in ("backend", "team", "urgency", "refund", "latency", "tokens in/out", "cost (USD)"):
        assert column in header
    rows = [line for line in out.splitlines() if re.match(r"\d+\s+(Jev|Claude)\s", line)]
    assert len(rows) == 2 * len(SAMPLE_TICKETS)


def test_the_threshold_option_changes_what_is_routed(capsys):
    assert "routed in code: 6 | escalated to the LLM: 0" in run(capsys, "--threshold", "0")[1]
    assert "routed in code: 0 | escalated to the LLM: 6" in run(capsys, "--threshold", "1")[1]


@pytest.mark.parametrize("value", ["1.5", "-0.1", "abc", "nan"])
def test_a_bad_threshold_is_rejected(value):
    with pytest.raises(SystemExit) as caught:
        triage.main(["--threshold", value])
    assert caught.value.code == 2


# --- Keys and live mode ----------------------------------------------------------------------------
def test_live_requires_both_keys_and_names_them_without_printing_values(capsys):
    with pytest.raises(SystemExit) as caught:
        triage.main(["--live"])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "TYPESAFE_API_KEY" in err and "ANTHROPIC_API_KEY" in err


def test_live_names_only_the_missing_key(capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder-typesafe")
    with pytest.raises(SystemExit):
        triage.main(["--live"])
    err = capsys.readouterr().err
    assert err.endswith("missing: ANTHROPIC_API_KEY\n")
    assert "placeholder-typesafe" not in err


def test_the_offline_flag_ignores_keys_and_never_builds_a_real_client(capsys, fake_keys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a real backend was constructed")

    monkeypatch.setattr(triage, "JevBackend", forbidden)
    monkeypatch.setattr(triage, "ClaudeBackend", forbidden)
    code, out, _ = run(capsys, "--offline")
    assert code == 0 and "Jev = mock (offline), Claude = mock (offline)" in out


def test_live_and_offline_cannot_be_combined():
    with pytest.raises(SystemExit) as caught:
        triage.main(["--live", "--offline"])
    assert caught.value.code == 2


def test_keys_in_the_environment_select_the_live_backends_and_show_measured_numbers(capsys, fake_keys, monkeypatch):
    monkeypatch.setattr(triage, "JevBackend", LiveJev)
    monkeypatch.setattr(triage, "ClaudeBackend", LiveClaude)
    code, out, _ = run(capsys, "--ticket", "any ticket")
    assert code == 0
    assert "Backends: Jev = live API, Claude = live API" in out
    assert "NOT measured" not in out
    assert "[142 ms, 118/12 tok, $0.000005]" in out


def test_compare_with_live_backends_shows_measured_numbers_and_agreement(capsys, fake_keys, monkeypatch):
    monkeypatch.setattr(triage, "JevBackend", LiveJev)
    monkeypatch.setattr(triage, "ClaudeBackend", LiveClaude)
    code, out, _ = run(capsys, "--compare", "--ticket", "any ticket")
    assert code == 0
    assert "142 ms" in out and "118 / 12" in out and "$0.000005" in out
    assert "1800 ms" in out and "400 / 90" in out and "$0.003000" in out
    assert "Jev (jev-test)" in out and "Claude (claude-test)" in out  # the model the API says served the call
    assert '"n/a"' not in out and '"-"' not in out  # the footnotes appear only when a cell needs them


def test_a_cost_the_sample_cannot_price_is_shown_as_n_a_with_a_note(capsys, fake_keys, monkeypatch):
    class FallbackClaude(LiveClaude):
        def decide(self, ticket):  # for example served by a fallback model with no published price here
            return Decision("billing", "today", True, stats=CallStats("other-model", 900.0, 400, 90, None))

    monkeypatch.setattr(triage, "JevBackend", LiveJev)
    monkeypatch.setattr(triage, "ClaudeBackend", FallbackClaude)
    _, out, _ = run(capsys, "--compare", "--ticket", "any ticket")
    assert re.search(r"Claude \(other-model\)\s+billing\s+today\s+yes\s+900 ms\s+400 / 90\s+n/a", out)
    assert '"n/a" means' in out


def test_one_key_gives_a_mixed_run(capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder-typesafe")
    monkeypatch.setattr(triage, "JevBackend", LiveJev)
    code, out, _ = run(capsys, "--ticket", "any ticket")
    assert code == 0 and "Backends: Jev = live API, Claude = mock (offline)" in out


# --- Inputs ----------------------------------------------------------------------------------------
def test_an_empty_ticket_is_rejected_at_the_command_line(capsys):
    with pytest.raises(SystemExit) as caught:
        triage.main(["--ticket", "  "])
    assert caught.value.code == 2
    assert "Ticket text is empty" in capsys.readouterr().err


def test_html_entities_and_emoji_are_shown_unchanged(capsys):
    ticket = "Refund &amp; credit &pound;49 \U0001F4B8 for my invoice payment"
    assert ticket in run(capsys, "--ticket", ticket)[1]


def test_terminal_escape_sequences_in_a_ticket_are_not_printed(capsys):
    ticket = "Refund please \x1b[31mRED\x1b[0m \x07 invoice payment"
    _, out, _ = run(capsys, "--ticket", ticket)
    assert "\x1b" not in out and "\x07" not in out


# --- Failures --------------------------------------------------------------------------------------
class Failing:
    def __init__(self, error):
        self.error = error

    def decide(self, ticket):
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        TypeSafeAuthenticationError(401, {"detail": "bad key"}, httpx2.Headers()),
        TypeSafeRateLimitError(429, None, httpx2.Headers()),
        TypeSafeAPIConnectionError("no route"),
        anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=SOME_URL), body=None),
        anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=SOME_URL), body=None),
        anthropic.APIConnectionError(request=SOME_URL),
    ],
    ids=lambda error: type(error).__name__ + "-" + type(error).__module__.split(".")[0],
)
def test_api_failures_from_either_sdk_exit_1_with_one_readable_line_and_no_traceback(capsys, monkeypatch, error):
    monkeypatch.setattr(triage, "build_backends", lambda offline=False: (Failing(error), MockClaude()))
    code, out, err = run(capsys, "--ticket", "invoice refund")
    assert code == 1
    assert err.startswith(f"error: {type(error).__name__}: ")
    assert err.count("\n") == 1 and "Traceback" not in err


class Refuser:
    def decide(self, ticket):
        raise Declined("cyber")

    def draft_reply(self, ticket):
        raise Declined("cyber")


def test_a_refusal_in_compare_mode_is_a_row_not_an_exception(capsys, monkeypatch):
    monkeypatch.setattr(triage, "build_backends", lambda offline=False: (MockJev(), Refuser()))
    code, out, _ = run(capsys, "--compare", "--ticket", "invoice refund payment")
    assert code == 0 and "declined (cyber)" in out


def test_a_refusal_while_drafting_prints_a_hand_over_note(capsys, monkeypatch):
    monkeypatch.setattr(triage, "build_backends", lambda offline=False: (MockJev(), Refuser()))
    code, out, _ = run(capsys, "--ticket", "Hello, can someone help me?")  # too vague to route, so it escalates
    assert code == 0
    assert "declined this request (category: cyber)" in out and "Hand this ticket to a person" in out


# --- Documentation -----------------------------------------------------------------------------------
@pytest.mark.parametrize(("marker", "args"), [("offline-output", []), ("offline-compare-output", ["--compare"])])
def test_the_readme_sample_output_is_a_real_run(capsys, marker, args):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    found = re.search(rf"<!-- {marker}:start -->\s*```text\n(.*?)\n```\s*<!-- {marker}:end -->", readme, re.S)
    assert found, f"README.md needs the sample output between <!-- {marker}:start --> and <!-- {marker}:end -->"
    assert found.group(1).rstrip() == run(capsys, *args)[1].rstrip()
