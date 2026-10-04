"""`python agent.py ...` as a user runs it: exit codes, messages, and what reaches stdout/stderr."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import read_outbox

import agent


def run_cli(*args: str, env: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    environment = {key: value for key, value in os.environ.items() if key not in drop}
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(agent.PROJECT_ROOT / "agent.py"), *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )


def test_claude_planner_without_an_api_key_exits_with_a_clear_message() -> None:
    result = run_cli("--planner", "claude", drop=("ANTHROPIC_API_KEY",))
    assert result.returncode == 2
    assert "ANTHROPIC_API_KEY" in result.stderr
    assert "--planner" in result.stderr or "rules" in result.stderr  # tells the user the way out
    assert "Traceback" not in result.stderr
    assert result.stdout == ""  # failed fast: no server was started, no trace printed


def test_claude_planner_with_an_empty_api_key_is_treated_as_missing() -> None:
    result = run_cli("--planner", "claude", env={"ANTHROPIC_API_KEY": ""})
    assert result.returncode == 2
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_default_run_uses_the_rules_planner_and_needs_no_key(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox.jsonl"
    result = run_cli(env={"EMAIL_OUTBOX_PATH": str(outbox)}, drop=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    for marker in ("Step 1 ", "Step 2 ", "Step 3 ", "Step 4 ", "Step 5 ", "Step 6 ", "Step 7 ", "Step 8 "):
        assert marker in result.stdout
    assert "tools/list" in result.stdout and "tools/call" in result.stdout
    assert "12 items (JSON array)" in result.stdout
    assert '"status": "sent"' in result.stdout
    assert "LLM " not in result.stdout  # no model involved
    (stored,) = read_outbox(outbox)
    assert stored["to"] == "ops-team"


def test_a_custom_request_is_taken_from_the_command_line(tmp_path: Path) -> None:
    result = run_cli(
        "Find the runbook for orders db failover",
        env={"EMAIL_OUTBOX_PATH": str(tmp_path / "outbox.jsonl")},
    )
    assert result.returncode == 0, result.stderr
    assert "Orders database: replica failover" in result.stdout
    assert "send_email" not in result.stdout.split("Step 4")[1]  # only the docs server was used
    assert not (tmp_path / "outbox.jsonl").exists()


def test_a_request_the_planner_cannot_handle_exits_1_with_an_error_line(tmp_path: Path) -> None:
    result = run_cli("What is the weather in Lisbon?", env={"EMAIL_OUTBOX_PATH": str(tmp_path / "outbox.jsonl")})
    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr and "ExceptionGroup" not in result.stderr
    assert "Step 3 " in result.stdout  # discovery had already happened


def test_help_describes_the_planner_option() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--planner" in result.stdout and "rules" in result.stdout and "claude" in result.stdout
