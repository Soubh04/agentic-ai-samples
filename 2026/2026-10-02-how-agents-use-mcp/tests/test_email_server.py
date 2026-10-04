"""send_email: nothing is delivered, everything lands in the outbox exactly as received."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import open_server, read_outbox, run_async

import agent
from servers import email_server

SPECIAL_SUBJECT = "Q&A <urgent> ✅"
SPECIAL_BODY = "Tom &amp; Jerry <b>bold</b> 🚨 café 日本語"


def send(**arguments: str):
    """Call send_email over a real MCP session (tests request the `outbox` fixture to redirect the file)."""

    async def call():
        async with open_server("email") as session:
            return await session.call_tool("send_email", arguments)

    return run_async(call())


def test_special_characters_are_stored_byte_for_byte(outbox: Path) -> None:
    result = send(to="ops-team", subject=SPECIAL_SUBJECT, body=SPECIAL_BODY)

    receipt = json.loads(result.content[0].text)
    assert receipt["status"] == "sent"

    raw = outbox.read_bytes()
    assert SPECIAL_BODY.encode("utf-8") in raw  # `&amp;`, `<b>` and the emoji are in the file untouched
    assert SPECIAL_SUBJECT.encode("utf-8") in raw
    assert b"\\u" not in raw  # written as UTF-8, not as \uXXXX escapes
    assert b"&lt;" not in raw and b"&amp;amp;" not in raw  # nothing was HTML-escaped on the way

    (stored,) = read_outbox(outbox)
    assert stored["body"] == SPECIAL_BODY
    assert stored["subject"] == SPECIAL_SUBJECT
    assert stored["to"] == "ops-team"
    assert stored["message_id"] == receipt["message_id"]


def test_awkward_content_round_trips_and_stays_on_one_line(outbox: Path) -> None:
    body = 'line1\nline2\r\n\ttabbed "quoted" back\\slash   é 👩‍💻 </script><!-- -->'
    send(to="a@example.com", subject="  padded subject  ", body=body)

    raw = outbox.read_bytes()
    assert raw.count(b"\n") == 1  # one message = one line, however many newlines the body has
    (stored,) = read_outbox(outbox)
    assert stored["body"] == body
    assert stored["subject"] == "  padded subject  "  # no trimming


@pytest.mark.parametrize("body", ["  indented", "trailing  ", "\n\nblank lines around\n\n", "\t", "   "])
def test_whitespace_in_the_body_is_kept_exactly(outbox: Path, body: str) -> None:
    send(to="ops-team", subject="whitespace", body=body)
    assert read_outbox(outbox)[0]["body"] == body  # never trimmed or normalised


def test_every_call_appends_a_new_line_with_its_own_message_id(outbox: Path) -> None:
    async def call_twice():
        async with open_server("email") as session:
            first = await session.call_tool("send_email", {"to": "ops-team", "subject": "one", "body": "1"})
            second = await session.call_tool("send_email", {"to": "ops-team", "subject": "two", "body": "2"})
            return first, second

    first, second = run_async(call_twice())
    ids = [json.loads(r.content[0].text)["message_id"] for r in (first, second)]
    assert ids[0] != ids[1]
    assert [m["subject"] for m in read_outbox(outbox)] == ["one", "two"]
    assert [m["message_id"] for m in read_outbox(outbox)] == ids


def test_an_empty_body_is_allowed(outbox: Path) -> None:
    result = send(to="ops-team", subject="No body", body="")
    assert not result.is_error
    assert read_outbox(outbox)[0]["body"] == ""


@pytest.mark.parametrize(
    "arguments",
    [
        {"to": "", "subject": "s", "body": "b"},
        {"to": "   ", "subject": "s", "body": "b"},
        {"to": "ops-team", "subject": "", "body": "b"},
        {"to": "ops-team", "subject": "\t", "body": "b"},
    ],
)
def test_blank_recipient_or_subject_is_a_readable_error_and_writes_nothing(outbox: Path, arguments: dict) -> None:
    result = send(**arguments)
    assert result.is_error
    assert re.search(r"must not be empty", result.content[0].text)
    assert not outbox.exists()


def test_missing_argument_is_a_readable_error(outbox: Path) -> None:
    result = send(to="ops-team", subject="s")  # no body
    assert result.is_error
    assert "body" in result.content[0].text
    assert not outbox.exists()


def test_outbox_location_comes_from_the_environment_and_defaults_inside_the_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(email_server.OUTBOX_ENV_VAR, raising=False)
    assert email_server.outbox_path() == email_server.DEFAULT_OUTBOX
    assert email_server.DEFAULT_OUTBOX == agent.PROJECT_ROOT / "outbox.jsonl"

    monkeypatch.setenv(email_server.OUTBOX_ENV_VAR, str(tmp_path / "custom.jsonl"))
    assert email_server.outbox_path() == tmp_path / "custom.jsonl"


def test_the_default_outbox_is_gitignored() -> None:
    ignored = (agent.PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert email_server.DEFAULT_OUTBOX.name in ignored


def test_the_email_server_has_no_way_to_send_real_mail() -> None:
    source = Path(email_server.__file__).read_text()
    for forbidden in ("smtplib", "socket", "requests", "httpx", "urllib", "subprocess"):
        assert not re.search(rf"^\s*(import|from)\s+{forbidden}\b", source, flags=re.MULTILINE), forbidden
