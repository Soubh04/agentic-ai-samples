"""Offline tests for agent_evals.py: no model, no network, no credentials."""

import json
from dataclasses import replace
from math import isclose, sqrt

import pytest

import agent_evals as ae

BOOK = '{"id": "b1", "goal": "book a flight", "action": "book", "expect": {"bookings": 1}, "reset": "fresh_sandbox"}'
CANCEL = (
    '{"id": "c1", "goal": "cancel a flight", "action": "cancel", "setup": {"bookings": 1}, '
    '"expect": {"bookings": 0}, "reset": "fresh_sandbox"}'
)


def one_task(line=BOOK):
    return ae.load_tasks([line])[0]


class ScriptedAgent:
    """Does the job only on the calls listed in `succeed_on` (1-based); every other call claims it without acting."""

    def __init__(self, succeed_on=(1,), extra_lookups=0):
        self.succeed_on = set(succeed_on)
        self.extra_lookups = extra_lookups
        self.calls = 0

    def run(self, task, env):
        self.calls += 1
        calls = [{"name": "find_booking", "arguments": {"task_id": task.id}, "result": {}}] * (1 + self.extra_lookups)
        if task.action == "book" and env.db.count("bookings", task.id) >= 1:
            return "Already done. Your flight is booked.", calls
        if self.calls in self.succeed_on:
            env.db.insert("bookings", task.id)
            calls = calls + [{"name": "book_flight", "arguments": {"task_id": task.id}, "result": {"ok": True}}]
        return "Your flight is booked.", calls


# ---- 1. tasks ----------------------------------------------------------------------------------------------


def test_the_task_file_loads_and_t07_matches_the_image():
    tasks = ae.read_tasks(ae.TASKS_FILE)
    assert len(tasks) == 8
    t07 = next(t for t in tasks if t.id == "t07")
    assert (t07.goal, t07.expect, t07.reset) == ("book LHR to JFK on 3 Nov", {"bookings": 1}, "fresh_sandbox")
    assert {t.scenario for t in tasks} == {"domestic", "international", "cancel"}


def test_defaults_blank_lines_and_setup():
    tasks = ae.load_tasks(["", BOOK, "   ", CANCEL])
    assert [t.id for t in tasks] == ["b1", "c1"]
    assert tasks[0].setup == {} and tasks[0].scenario == "b1"
    assert tasks[1].setup == {"bookings": 1}


@pytest.mark.parametrize(
    "line, message",
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"id": "x", "goal": "g", "action": "book", "expect": {"bookings": 1}}', "missing reset"),
        (BOOK.replace('"b1"', '"  "'), "id and goal must not be empty"),
        (BOOK.replace('"book"', '"refund"'), "action must be one of"),
        (BOOK.replace("fresh_sandbox", "reused"), "reset must be one of"),
        (BOOK.replace('{"bookings": 1}', "{}"), "expect must map"),
        (BOOK.replace('{"bookings": 1}', '{"bookings": -1}'), "expect must map"),
        (BOOK.replace('{"bookings": 1}', '{"bookings": true}'), "expect must map"),
        (BOOK.replace('{"bookings": 1}', '{"bookings": "1"}'), "expect must map"),
        (CANCEL.replace('"setup": {"bookings": 1}', '"setup": [1]'), "setup must map"),
    ],
)
def test_invalid_lines_are_rejected_with_the_line_number(line, message):
    with pytest.raises(ae.TaskFileError, match=f"line 1: {message}"):
        ae.load_tasks([line])


def test_duplicate_ids_and_an_empty_file_are_rejected(tmp_path):
    with pytest.raises(ae.TaskFileError, match="line 2: duplicate id b1"):
        ae.load_tasks([BOOK, BOOK])
    with pytest.raises(ae.TaskFileError, match="no tasks"):
        ae.load_tasks(["", " "])
    with pytest.raises(ae.TaskFileError, match="cannot read"):
        ae.read_tasks(tmp_path / "missing.jsonl")
    latin1 = tmp_path / "latin1.jsonl"
    latin1.write_bytes(b"\xff\n")
    with pytest.raises(ae.TaskFileError, match="not UTF-8"):
        ae.read_tasks(latin1)


# ---- 2. outcome --------------------------------------------------------------------------------------------


def test_a_claim_without_a_booking_fails_the_outcome_grade_but_fools_the_reply_grader():
    task = one_task()
    env = ae.new_env(task)
    reply, _ = ScriptedAgent(succeed_on=()).run(task, env)
    assert ae.grade_reply(reply) is True
    assert ae.grade(task, env) is False


def test_the_outcome_grade_reads_the_sandbox_for_book_and_cancel():
    book, cancel = ae.load_tasks([BOOK, CANCEL])
    env = ae.new_env(book)
    env.db.insert("bookings", book.id)
    assert ae.grade(book, env)
    env.db.insert("bookings", book.id)
    assert not ae.grade(book, env)  # two bookings is not the expected end state
    env = ae.new_env(cancel)
    assert env.db.count("bookings", cancel.id) == 1 and not ae.grade(cancel, env)
    assert env.db.delete("bookings", cancel.id) and ae.grade(cancel, env)
    assert not env.db.delete("bookings", cancel.id)


def test_the_outcome_grade_checks_every_table_in_expect():
    task = replace(one_task(), expect={"bookings": 1, "refunds": 0})
    env = ae.new_env(task)
    env.db.insert("bookings", task.id)
    assert ae.grade(task, env)
    env.db.insert("refunds", task.id)
    assert not ae.grade(task, env)  # the booking is right, the extra refund is not


def test_the_reply_grader_trusts_both_claims():
    assert ae.grade_reply("Your flight is booked.")
    assert ae.grade_reply("Your booking is cancelled.")
    assert not ae.grade_reply("Sorry, the booking system did not respond. Please try again.")


def test_the_outcome_grade_accepts_any_valid_tool_path():
    task = one_task()
    short = ae.run_task(task, ScriptedAgent(), 1)[0]
    long = ae.run_task(task, ScriptedAgent(extra_lookups=2), 1)[0]
    assert len(long.tool_calls) > len(short.tool_calls)
    assert short.passed and long.passed


def test_a_shared_sandbox_inflates_the_score():
    task = one_task()
    fresh = ae.run_task(task, ScriptedAgent(succeed_on=(1,)), 5)
    shared = ae.run_task(replace(task, reset="shared_sandbox"), ScriptedAgent(succeed_on=(1,)), 5)
    assert sum(t.passed for t in fresh) == 1
    assert sum(t.passed for t in shared) == 5  # trials 2 to 5 see the booking trial 1 left behind


def test_the_fake_agent_outcomes():
    task = one_task()
    always = ae.run_task(task, ae.FakeAgent(1, {"b1": 1.0}), 6)
    assert all(t.passed for t in always)
    assert all(t.tool_calls[-1]["name"] == "book_flight" for t in always)
    claims = ae.run_task(task, ae.FakeAgent(1, {"b1": 0.0}, claim_rate=1.0), 6)
    assert not any(t.passed for t in claims) and all(ae.grade_reply(t.reply) for t in claims)
    errors = ae.run_task(task, ae.FakeAgent(1, {"b1": 0.0}, claim_rate=0.0), 6)
    assert all(t.tool_calls[-1]["result"] == {"error": "timeout"} for t in errors)
    assert not any(t.passed or ae.grade_reply(t.reply) for t in errors)


def test_the_fake_agent_stops_when_the_job_already_looks_done():
    book, cancel = ae.load_tasks([BOOK, CANCEL])
    env = ae.new_env(book)
    env.db.insert("bookings", book.id)
    reply, calls = ae.FakeAgent(1, {"b1": 0.0}).run(book, env)
    assert reply.startswith("Already done") and [c["name"] for c in calls] == ["find_booking"]
    env = ae.new_env(cancel)
    env.db.delete("bookings", cancel.id)
    reply, calls = ae.FakeAgent(1, {"c1": 0.0}).run(cancel, env)
    assert reply.startswith("Already done") and [c["name"] for c in calls] == ["find_booking"]
    env = ae.new_env(cancel)  # the booking still exists, so the agent has to act
    reply, calls = ae.FakeAgent(1, {"c1": 1.0}).run(cancel, env)
    assert [c["name"] for c in calls] == ["find_booking", "cancel_flight"] and ae.grade(cancel, env)


def test_evaluate_is_deterministic_and_the_demo_shows_inflation():
    tasks = ae.read_tasks(ae.TASKS_FILE)
    first = ae.evaluate(tasks, ae.AGENT_V1, 8, 7)
    assert first == ae.evaluate(tasks, ae.AGENT_V1, 8, 7)
    shared = ae.evaluate([replace(t, reset="shared_sandbox") for t in tasks], ae.AGENT_V1, 8, 7)
    passes = lambda results: sum(t.passed for trials in results.values() for t in trials)  # noqa: E731
    assert passes(shared) > passes(first)
    with pytest.raises(ValueError):
        ae.run_task(tasks[0], ScriptedAgent(), 0)


# ---- 3. process: Foundry -----------------------------------------------------------------------------------


def test_the_tool_use_quality_criterion_matches_the_documented_shape():
    assert ae.tool_use_quality_criterion("my-judge") == {
        "type": "azure_ai_evaluator",
        "name": "tool_use_quality",
        "evaluator_name": "builtin.tool_use_quality",
        "initialization_parameters": {"deployment_name": "my-judge", "evaluation_level": "turn"},
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    }
    assert ae.tool_use_quality_criterion(ae.DEFAULT_JUDGE_DEPLOYMENT)["initialization_parameters"][
        "deployment_name"
    ] == "gpt-5.6-luna"
    with pytest.raises(ValueError):
        ae.tool_use_quality_criterion("  ")


def test_a_foundry_row_pairs_every_tool_call_with_its_result():
    task = one_task()
    trial = ae.run_task(task, ae.FakeAgent(1, {"b1": 0.0}, claim_rate=0.0), 1)[0]  # find, then a timeout
    row = ae.foundry_row(task, trial)
    assert row["query"] == [
        {"role": "system", "content": ae.SYSTEM_PROMPT},
        {"role": "user", "content": "book a flight"},
    ]
    calls = [m for m in row["response"] if m["role"] == "assistant" and isinstance(m["content"], list)]
    results = [m for m in row["response"] if m["role"] == "tool"]
    assert [c["content"][0]["tool_call_id"] for c in calls] == [r["tool_call_id"] for r in results] == ["call_1", "call_2"]
    assert results[-1]["content"][0] == {"type": "tool_result", "tool_result": {"error": "timeout"}}
    assert row["response"][-1] == {"role": "assistant", "content": trial.reply}
    defined = {d["name"] for d in row["tool_definitions"]}
    assert {c["content"][0]["name"] for c in calls} <= defined
    assert all(d["parameters"]["type"] == "object" for d in row["tool_definitions"])
    json.dumps(row)  # a dataset row must serialise to JSON Lines


# ---- 4. judge ----------------------------------------------------------------------------------------------


def test_the_swap_test_turns_position_bias_into_a_tie():
    assert ae.swap_test(ae.first_answer_judge, "q", "x answer", "y answer") == "tie"


def test_the_swap_test_keeps_a_consistent_verdict():
    judge = ae.quality_judge({"good": 0.9, "weak": 0.4, "same": 0.4})
    assert ae.swap_test(judge, "q", "good", "weak") == "x"
    assert ae.swap_test(judge, "q", "weak", "good") == "y"
    assert ae.swap_test(judge, "q", "weak", "same") == "tie"
    assert [ae.flip(v) for v in ("A", "B", "tie")] == ["B", "A", "tie"]
    with pytest.raises(ValueError, match="verdict"):
        ae.swap_test(lambda q, a, b: "first", "q", "good", "weak")


def test_cohen_kappa():
    assert isclose(ae.cohen_kappa(ae.JUDGE_LABELS, ae.HUMAN_LABELS), 0.625)  # (10/12 - 5/9) / (1 - 5/9)
    assert ae.cohen_kappa(["p", "f", "p"], ["p", "f", "p"]) == 1.0
    assert ae.cohen_kappa(["p", "p"], ["p", "p"]) == 1.0
    assert ae.cohen_kappa(["p", "p", "f", "f"], ["p", "f", "p", "f"]) == 0.0  # agreement no better than chance
    for a, b in (([], []), (["p"], ["p", "f"])):
        with pytest.raises(ValueError):
            ae.cohen_kappa(a, b)


# ---- 5. pass@k and pass^k ----------------------------------------------------------------------------------


def test_the_worked_example_in_the_image():
    assert ae.pass_at_k(8, 6, 4) == 1.0
    assert isclose(ae.pass_hat_k(8, 6, 4), 15 / 70)
    assert round(ae.pass_hat_k(8, 6, 4), 2) == 0.21


def test_estimator_edge_cases():
    assert ae.pass_at_k(8, 0, 4) == ae.pass_hat_k(8, 0, 4) == 0.0
    assert ae.pass_at_k(8, 8, 4) == ae.pass_hat_k(8, 8, 4) == 1.0
    assert ae.pass_at_k(8, 5, 1) == ae.pass_hat_k(8, 5, 1) == 5 / 8
    for n, c, k in ((0, 0, 1), (8, 9, 4), (8, -1, 4), (8, 6, 0), (8, 6, 9)):
        with pytest.raises(ValueError):
            ae.pass_at_k(n, c, k)
        with pytest.raises(ValueError):
            ae.pass_hat_k(n, c, k)


def test_pass_hat_k_never_exceeds_pass_at_k():
    for n in range(1, 11):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert ae.pass_hat_k(n, c, k) <= ae.pass_at_k(n, c, k) + 1e-12


def test_pass_hat_k_matches_the_post_and_image_figures():
    # With many trials the estimator approaches p^k: 0.75^3 in the post, 0.9^10 in the image.
    assert isclose(ae.pass_hat_k(4000, 3000, 3), 0.75**3, abs_tol=0.002)
    assert round(0.75**3, 2) == 0.42
    assert isclose(ae.pass_hat_k(10000, 9000, 10), 0.9**10, abs_tol=0.002)
    assert round(0.9**10, 2) == 0.35


# ---- 6. gate -----------------------------------------------------------------------------------------------


def test_bernoulli_interval_figures_from_the_post_and_image():
    assert round(ae.Z_95 * ae.se_bernoulli(0.75, 100), 4) == 0.0849  # image: ±8.5 points
    assert round(ae.Z_95 * ae.se_bernoulli(0.75, 30), 3) == 0.155  # post: about ±15 points
    assert isclose(ae.se_bernoulli(0.75, 400), ae.se_bernoulli(0.75, 100) / 2)  # image: 4× tasks halves it
    # post: a 5-point gain between two unpaired runs of 30 tasks sits well inside the noise (about ±22 points)
    assert round(ae.Z_95 * sqrt(2) * ae.se_bernoulli(0.75, 30), 2) == 0.22
    for p, n in ((-0.1, 10), (1.1, 10), (0.5, 0)):
        with pytest.raises(ValueError):
            ae.se_bernoulli(p, n)


def test_se_clt():
    assert ae.se_clt([0, 1]) == 0.5
    scores = [1] * 75 + [0] * 25
    assert isclose(ae.se_clt(scores), ae.se_bernoulli(0.75, 100) * sqrt(100 / 99))
    with pytest.raises(ValueError):
        ae.se_clt([1.0])


def test_se_clustered():
    scores = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert isclose(ae.se_clustered(scores, ["a", "b", "c", "d", "e", "f"]), ae.se_clt(scores))
    together = ["s1", "s1", "s2", "s2", "s3", "s3"]
    paired_scores = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    # By hand from Miller equation 4: deviations 1/3, 1/3, -2/3, -2/3, 1/3, 1/3 around the mean 2/3.
    # SE_CLT^2 = (4/3) / 5 / 6 = 4/90; within-cluster cross terms 2/9 + 8/9 + 2/9 = 12/9, divided by n^2 = 36.
    assert isclose(ae.se_clt(paired_scores) ** 2, 4 / 90)
    assert isclose(ae.se_clustered(paired_scores, together), sqrt(4 / 90 + (12 / 9) / 36))
    assert round(ae.se_clustered(paired_scores, together), 5) == 0.28545
    # Negative within-cluster correlation shrinks the standard error, and it stays above zero.
    alternating = [1.0, 0.0, 1.0, 0.0]
    assert 0.0 < ae.se_clustered(alternating, ["s1", "s1", "s2", "s2"]) < ae.se_clt(alternating)
    with pytest.raises(ValueError):
        ae.se_clustered([1.0, 0.0], ["s1"])


def test_paired_differences_shrink_the_error_bar():
    v1 = [0.2, 0.9, 0.5, 0.1, 0.7, 0.4]
    v2 = [0.3, 1.0, 0.6, 0.15, 0.8, 0.5]
    assert isclose(ae.se_paired(v2, v1), ae.se_clt([b - a for a, b in zip(v1, v2)]))
    assert isclose(ae.se_unpaired(v2, v1), sqrt(ae.se_clt(v2) ** 2 + ae.se_clt(v1) ** 2))
    assert ae.se_paired(v2, v1) < ae.se_unpaired(v2, v1) / 10
    with pytest.raises(ValueError):
        ae.se_paired([1.0, 0.0], [1.0])


def test_the_gate_uses_the_lower_bound():
    assert isclose(ae.lower_bound(0.5, 0.1), 0.5 - 0.196)
    assert ae.gate(0.25, 0.25) == 0
    assert ae.gate(0.2499, 0.25) == 1
    assert ae.gate(ae.lower_bound(0.3, 0.05), 0.25) == 1  # the mean clears the target, the lower bound does not


def test_the_opentelemetry_event_uses_the_semantic_convention_names():
    event = ae.evaluation_event("pass^4", 0.28, "fail", "95% lower bound 0.01", "resp_1")
    assert event == {
        "name": "gen_ai.evaluation.result",
        "attributes": {
            "gen_ai.evaluation.name": "pass^4",
            "gen_ai.evaluation.score.value": 0.28,
            "gen_ai.evaluation.score.label": "fail",
            "gen_ai.evaluation.explanation": "95% lower bound 0.01",
            "gen_ai.response.id": "resp_1",
        },
    }
    assert ae.evaluation_event("tool_use_quality", label="pass")["attributes"] == {
        "gen_ai.evaluation.name": "tool_use_quality",
        "gen_ai.evaluation.score.label": "pass",
    }
    with pytest.raises(ValueError):
        ae.evaluation_event(" ")


# ---- the command line --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [["--trials", "0"], ["--k", "9"], ["--k", "0"], ["--target", "1.5"], ["--kappa-min", "2"], ["--trials", "x"]],
)
def test_invalid_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        ae.main(argv)
    assert exc.value.code == 2


def test_task_file_problems_exit_2(tmp_path, capsys):
    assert ae.main(["--tasks", str(tmp_path / "missing.jsonl")]) == 2
    single = tmp_path / "one.jsonl"
    single.write_text(BOOK + "\n", encoding="utf-8")
    assert ae.main(["--tasks", str(single)]) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text(BOOK + "\n{oops\n", encoding="utf-8")
    assert ae.main(["--tasks", str(bad)]) == 2
    assert "line 2: not valid JSON" in capsys.readouterr().err
    latin1 = tmp_path / "latin1.jsonl"
    latin1.write_bytes(b"\xff\n")
    assert ae.main(["--tasks", str(latin1)]) == 2
    assert "not UTF-8" in capsys.readouterr().err


def test_the_default_run_fails_the_pass4_gate_and_pass1_passes(capsys):
    assert ae.main([]) == 1  # 8 tasks: the interval is too wide for pass^4 to clear the target
    out = capsys.readouterr().out
    for heading in ("== 1. Tasks", "== 2. Outcome", "== 3. Process", "== 4. Judge", "== 5. pass@k", "== 6. Gate"):
        assert heading in out
    assert '"name": "gen_ai.evaluation.result"' in out
    assert "no cloud call" in out
    assert ae.main(["--k", "1"]) == 0
    assert ae.main(["--k", "4", "--target", "0"]) == 1  # the lower bound is below 0: too few tasks to pass at all


def test_the_default_run_prints_what_the_readme_shows(capsys):
    assert ae.main([]) == 1
    out = capsys.readouterr().out
    for line in (
        "64 trials. The reply says it is done in 51; the sandbox agrees in 45.",
        "The outcome grader caught 6 replies that claim a change the sandbox does not show.",
        "One shared sandbox per task instead of a fresh one per trial: 59 pass instead of 45.",
        "Swap test, position-biased judge: tie",
        "Swap test, quality judge: x (the better reply)",
        "Cohen's kappa, judge vs human on 12 replies: 0.63 (minimum 0.60): calibrated",
        "mean                          0.99    0.28",
        "Mean pass^4 0.28 over 8 tasks. SE plain 0.117, clustered by scenario 0.144.",
        "95% lower bound -0.003, target 0.25: fail, exit code 1.",
        "Agent v2 vs v1 on the same tasks: +0.11 pass^4; SE paired 0.066, unpaired 0.186.",
    ):
        assert line in out
    event = json.loads(out.split("OpenTelemetry event: ")[1])
    assert event["attributes"]["gen_ai.evaluation.score.label"] == "fail"
    assert event["attributes"]["gen_ai.evaluation.score.value"] == 0.2786


def test_one_run_per_task_and_one_average_passes_the_same_gate(capsys):
    # The post's mistake: one trial per task gives a green tick for the agent that fails at pass^4 above.
    assert ae.main(["--trials", "1", "--k", "1"]) == 0
    assert "95% lower bound 0.283, target 0.25: pass, exit code 0." in capsys.readouterr().out


def test_an_uncalibrated_judge_is_reported(capsys):
    ae.main(["--kappa-min", "0.7"])
    assert "0.63 (minimum 0.70): not calibrated" in capsys.readouterr().out
