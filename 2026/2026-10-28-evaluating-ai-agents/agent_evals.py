"""Evaluate an AI agent in six blocks, in one small offline script.

The six blocks from the LinkedIn post "Evaluating AI agents: what to measure before you ship":
  tasks     one JSON line per task: goal, expected end state, reset rule (tasks.jsonl)
  outcome   grade the rows the agent left in its sandbox; the agent's reply is never an input
  process   Microsoft Foundry's Tool Use Quality evaluator (preview): the testing criterion and one dataset row
  judge     a position-swap test, then Cohen's kappa between judge labels and human labels
  pass^k    unbiased pass@k and pass^k estimators from n trials per task
  gate      standard errors (plain, clustered, paired), the 95% lower bound and the CI exit code

The agent is a seeded fake: each trial it does the job, says it did without doing it, or hits an error.
No model, no network, no cloud service.

    uv run python agent_evals.py
    uv run python agent_evals.py --trials 8 --k 4 --target 0.5

Exit codes: 0 the gate passed, 1 the gate failed, 2 invalid argument or task file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, replace
from math import comb, sqrt
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

Z_95 = 1.96
TASKS_FILE = Path(__file__).with_name("tasks.jsonl")
ACTIONS = ("book", "cancel")
RESETS = ("fresh_sandbox", "shared_sandbox")
DEFAULT_TRIALS = 8
DEFAULT_K = 4
DEFAULT_TARGET = 0.25  # the author's demo value: set your own from what the agent must deliver
DEFAULT_KAPPA_MIN = 0.6  # the author's demo value, not a standard: set it from your own risk
# Deployment names are yours to choose. For composite evaluators Microsoft recommends a GPT-5.6-family judge and
# names gpt-5.6-luna as the lowest-cost option that keeps evaluation quality high.
DEFAULT_JUDGE_DEPLOYMENT = "gpt-5.6-luna"

# Chance per scenario that one trial does the job. v2 is the "improved" agent for the paired comparison.
AGENT_V1 = {"domestic": 0.85, "international": 0.6, "cancel": 0.75}
AGENT_V2 = {"domestic": 0.9, "international": 0.75, "cancel": 0.8}


class TaskFileError(ValueError):
    """The task file is missing, empty or has an invalid line."""


# ---- 1. tasks: one JSON line per task ---------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    id: str
    goal: str
    action: str  # "book" or "cancel"
    scenario: str  # tasks that share a scenario form one cluster for the standard error
    setup: dict[str, int]  # rows the sandbox holds for this task before the agent starts
    expect: dict[str, int]  # rows the sandbox must hold for this task when the agent is done
    reset: str  # "fresh_sandbox": a new sandbox per trial; "shared_sandbox": one sandbox for every trial


def _counts(value: object, number: int, name: str, allow_empty: bool) -> dict[str, int]:
    valid = isinstance(value, dict) and (allow_empty or value) and all(
        isinstance(k, str) and k and isinstance(v, int) and not isinstance(v, bool) and v >= 0 for k, v in value.items()
    )
    if not valid:
        raise TaskFileError(f"line {number}: {name} must map table names to row counts of 0 or more")
    return dict(value)  # type: ignore[arg-type]


def load_tasks(lines: Iterable[str]) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskFileError(f"line {number}: not valid JSON ({exc.msg})") from None
        if not isinstance(raw, dict):
            raise TaskFileError(f"line {number}: expected a JSON object")
        missing = [key for key in ("id", "goal", "action", "expect", "reset") if key not in raw]
        if missing:
            raise TaskFileError(f"line {number}: missing {', '.join(missing)}")
        task_id, goal = str(raw["id"]).strip(), str(raw["goal"]).strip()
        if not task_id or not goal:
            raise TaskFileError(f"line {number}: id and goal must not be empty")
        if task_id in seen:
            raise TaskFileError(f"line {number}: duplicate id {task_id}")
        if raw["action"] not in ACTIONS:
            raise TaskFileError(f"line {number}: action must be one of {', '.join(ACTIONS)}")
        if raw["reset"] not in RESETS:
            raise TaskFileError(f"line {number}: reset must be one of {', '.join(RESETS)}")
        seen.add(task_id)
        tasks.append(
            Task(
                id=task_id,
                goal=goal,
                action=raw["action"],
                scenario=str(raw.get("scenario") or task_id),
                setup=_counts(raw.get("setup", {}), number, "setup", allow_empty=True),
                expect=_counts(raw["expect"], number, "expect", allow_empty=False),
                reset=raw["reset"],
            )
        )
    if not tasks:
        raise TaskFileError("the task file has no tasks")
    return tasks


def read_tasks(path: Path) -> list[Task]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskFileError(f"cannot read {path}: {exc.strerror}") from None
    except UnicodeDecodeError:
        raise TaskFileError(f"{path} is not UTF-8 text") from None
    return load_tasks(text.splitlines())


# ---- the sandbox the agent changes ------------------------------------------------------------------------


class Sandbox:
    """An in-memory stand-in for the system the agent changes: rows per table, each tagged with a task id."""

    def __init__(self) -> None:
        self.tables: dict[str, list[str]] = {}

    def insert(self, table: str, task_id: str) -> None:
        self.tables.setdefault(table, []).append(task_id)

    def delete(self, table: str, task_id: str) -> bool:
        rows = self.tables.get(table, [])
        if task_id in rows:
            rows.remove(task_id)
            return True
        return False

    def count(self, table: str, task_id: str) -> int:
        return self.tables.get(table, []).count(task_id)


@dataclass
class Env:
    db: Sandbox


def new_env(task: Task) -> Env:
    env = Env(Sandbox())
    for table, rows in task.setup.items():
        for _ in range(rows):
            env.db.insert(table, task.id)
    return env


# ---- the agent under test: a seeded fake ------------------------------------------------------------------


class Agent(Protocol):
    def run(self, task: Task, env: Env) -> tuple[str, list[dict]]: ...


class FakeAgent:
    """A seeded stand-in for an LLM agent. Each trial it does the job, claims it without doing it, or hits an error.

    Like a careful real agent, it first looks for an existing booking and stops if the job already looks done. In a
    shared sandbox that check sees rows an earlier trial left behind, which is how shared state inflates scores.
    """

    def __init__(self, seed: int | str, reliability: dict[str, float], claim_rate: float = 0.5) -> None:
        self.rng = random.Random(seed)
        self.reliability = reliability  # chance per scenario that a trial does the job
        self.claim_rate = claim_rate  # chance that a trial that does not do the job still says it did

    def run(self, task: Task, env: Env) -> tuple[str, list[dict]]:
        found = env.db.count("bookings", task.id)
        calls = [{"name": "find_booking", "arguments": {"task_id": task.id}, "result": {"bookings": found}}]
        done = "Your flight is booked." if task.action == "book" else "Your booking is cancelled."
        if (found >= 1) if task.action == "book" else (found == 0):
            return "Already done. " + done, calls
        roll = self.rng.random()
        chance = self.reliability.get(task.scenario, 0.5)
        if roll < chance:
            if task.action == "book":
                env.db.insert("bookings", task.id)
            else:
                env.db.delete("bookings", task.id)
            calls.append({"name": f"{task.action}_flight", "arguments": {"task_id": task.id}, "result": {"ok": True}})
            return done, calls
        if roll < chance + (1 - chance) * self.claim_rate:
            return done, calls  # the claim with nothing behind it: no write was ever made
        calls.append({"name": f"{task.action}_flight", "arguments": {"task_id": task.id}, "result": {"error": "timeout"}})
        return "Sorry, the booking system did not respond. Please try again.", calls


# ---- 2. outcome: grade the sandbox, not the reply ---------------------------------------------------------


def grade(task: Task, env: Env) -> bool:
    """Outcome grading: compare the rows the agent left with the rows the task expects. The reply is not an input."""
    return all(env.db.count(table, task.id) == rows for table, rows in task.expect.items())


def grade_reply(reply: str) -> bool:
    """The mistake the post warns about: trusting what the agent says it did."""
    text = reply.lower()
    return "booked" in text or "cancelled" in text


@dataclass(frozen=True)
class Trial:
    reply: str
    tool_calls: tuple[dict, ...]
    passed: bool  # the outcome grade


def run_task(task: Task, agent: Agent, trials: int) -> list[Trial]:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    shared = new_env(task) if task.reset == "shared_sandbox" else None
    results = []
    for _ in range(trials):
        env = shared if shared is not None else new_env(task)
        reply, calls = agent.run(task, env)
        results.append(Trial(reply, tuple(calls), grade(task, env)))
    return results


def evaluate(tasks: Sequence[Task], reliability: dict[str, float], trials: int, seed: int) -> dict[str, list[Trial]]:
    # One random stream per task, so two agent versions run with the same seed see the same draws: a paired design.
    return {t.id: run_task(t, FakeAgent(f"{seed}/{t.id}", reliability), trials) for t in tasks}


# ---- 3. process: Microsoft Foundry Tool Use Quality (preview) ---------------------------------------------

SYSTEM_PROMPT = "You are a travel booking agent. Book or cancel flights with the tools provided."
TOOL_DEFINITIONS = [
    {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"task_id": {"type": "string", "description": "The booking request id."}},
            "required": ["task_id"],
        },
    }
    for name, description in (
        ("find_booking", "Count the bookings that exist for a request."),
        ("book_flight", "Create the booking for a request."),
        ("cancel_flight", "Cancel the booking for a request."),
    )
]


def tool_use_quality_criterion(deployment_name: str) -> dict:
    """The testing criterion for Foundry's composite Tool Use Quality evaluator (preview), at turn level.

    It batches Tool Call Accuracy, Tool Call Success, Tool Input Accuracy, Tool Output Utilization and Tool Selection
    into one LLM judge call. Shape from
    https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/agent-evaluators
    """
    if not deployment_name.strip():
        raise ValueError("deployment_name is required")
    return {
        "type": "azure_ai_evaluator",
        "name": "tool_use_quality",
        "evaluator_name": "builtin.tool_use_quality",
        "initialization_parameters": {"deployment_name": deployment_name, "evaluation_level": "turn"},
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "tool_definitions": "{{item.tool_definitions}}",
        },
    }


def foundry_row(task: Task, trial: Trial) -> dict:
    """One dataset row in the query, response and tool_definitions shape the Foundry agent evaluators read."""
    response: list[dict] = []
    for number, call in enumerate(trial.tool_calls, start=1):
        call_id = f"call_{number}"
        response.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "tool_call_id": call_id, "name": call["name"], "arguments": call["arguments"]}
                ],
            }
        )
        response.append(
            {"role": "tool", "tool_call_id": call_id, "content": [{"type": "tool_result", "tool_result": call["result"]}]}
        )
    response.append({"role": "assistant", "content": trial.reply})
    return {
        "query": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.goal}],
        "response": response,
        "tool_definitions": TOOL_DEFINITIONS,
    }


# ---- 4. judge: swap test and agreement with humans --------------------------------------------------------

Judge = Callable[[str, str, str], str]  # (question, first answer, second answer) -> "A", "B" or "tie"
VERDICTS = ("A", "B", "tie")


def flip(verdict: str) -> str:
    return {"A": "B", "B": "A"}.get(verdict, verdict)


def swap_test(judge: Judge, question: str, x: str, y: str) -> str:
    """Judge both orders. Returns "x" or "y" when both verdicts agree, and "tie" when they do not (position bias)."""
    a = judge(question, x, y)
    b = judge(question, y, x)
    for verdict in (a, b):
        if verdict not in VERDICTS:
            raise ValueError(f"a judge verdict must be one of {', '.join(VERDICTS)}, got {verdict!r}")
    win = a if a == flip(b) else "tie"
    return {"A": "x", "B": "y"}.get(win, "tie")


def first_answer_judge(question: str, first: str, second: str) -> str:
    """A judge with pure position bias: it prefers whichever answer it is shown first."""
    return "A"


def quality_judge(quality: dict[str, float]) -> Judge:
    """A judge that compares a hidden quality score, so the order it sees the answers in does not matter."""

    def judge(question: str, first: str, second: str) -> str:
        a, b = quality[first], quality[second]
        return "A" if a > b else "B" if b > a else "tie"

    return judge


def cohen_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Agreement between two raters, corrected for the agreement their label rates would give by chance."""
    if not rater_a or len(rater_a) != len(rater_b):
        raise ValueError("both raters must label the same, non-empty list of items")
    n = len(rater_a)
    observed = sum(a == b for a, b in zip(rater_a, rater_b)) / n
    labels = set(rater_a) | set(rater_b)
    expected = sum((list(rater_a).count(label) / n) * (list(rater_b).count(label) / n) for label in labels)
    if expected == 1:
        return 1.0  # both raters used one and the same label for every item
    return (observed - expected) / (1 - expected)


# ---- 5. pass@k and pass^k ---------------------------------------------------------------------------------


def _check_nck(n: int, c: int, k: int) -> None:
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need n >= 1, 0 <= c <= n and 1 <= k <= n (got n={n}, c={c}, k={k})")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k trials, drawn from the n you ran (c passed), passed. Unbiased (Chen et al.)."""
    _check_nck(n, c, k)
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: chance that all k trials, drawn from the n you ran (c passed), passed. Unbiased (Yao et al.)."""
    _check_nck(n, c, k)
    return comb(c, k) / comb(n, k)


# ---- 6. gate: error bars, then an exit code ---------------------------------------------------------------


def se_bernoulli(p: float, n: int) -> float:
    """Standard error of a pass rate p over n pass/fail scores (Miller, 2024, equation 2)."""
    if not 0 <= p <= 1 or n < 1:
        raise ValueError("need 0 <= p <= 1 and n >= 1")
    return sqrt(p * (1 - p) / n)


def _mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores)


def se_clt(scores: Sequence[float]) -> float:
    """Standard error of the mean score, any score scale (Miller, 2024, equation 1)."""
    if len(scores) < 2:
        raise ValueError("need at least 2 scores for a standard error")
    mean = _mean(scores)
    variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
    return sqrt(variance / len(scores))


def se_clustered(scores: Sequence[float], clusters: Sequence[str]) -> float:
    """Clustered standard error (Miller, 2024, equation 4): adds the covariance of scores that share a cluster."""
    if len(scores) != len(clusters):
        raise ValueError("need one cluster label per score")
    n, mean = len(scores), _mean(scores) if scores else 0.0
    groups: dict[str, list[float]] = {}
    for score, cluster in zip(scores, clusters):
        groups.setdefault(cluster, []).append(score - mean)
    # sum over i and j != i inside a cluster of d_i * d_j, which is (sum of d)^2 - sum of d^2
    cross = sum(sum(devs) ** 2 - sum(d * d for d in devs) for devs in groups.values())
    return sqrt(se_clt(scores) ** 2 + cross / n**2)


def se_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard error of the mean difference on the same tasks (Miller, 2024, equation 7)."""
    if len(a) != len(b):
        raise ValueError("a paired comparison needs both versions scored on the same tasks")
    return se_clt([x - y for x, y in zip(a, b)])


def se_unpaired(a: Sequence[float], b: Sequence[float]) -> float:
    """The same difference treated as two independent samples, which ignores that the tasks are shared."""
    return sqrt(se_clt(a) ** 2 + se_clt(b) ** 2)


def lower_bound(mean: float, se: float, z: float = Z_95) -> float:
    return mean - z * se


def gate(low: float, target: float) -> int:
    """The CI exit code: 0 when the 95% lower bound reaches the target, 1 when it does not."""
    return 0 if low >= target else 1


def evaluation_event(
    name: str,
    value: float | None = None,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
) -> dict:
    """Attributes of the OpenTelemetry `gen_ai.evaluation.result` event (GenAI semantic conventions, Development)."""
    if not name.strip():
        raise ValueError("gen_ai.evaluation.name is required")
    attributes: dict[str, object] = {"gen_ai.evaluation.name": name}
    if value is not None:
        attributes["gen_ai.evaluation.score.value"] = float(value)
    if label is not None:
        attributes["gen_ai.evaluation.score.label"] = label
    if explanation:
        attributes["gen_ai.evaluation.explanation"] = explanation
    if response_id:
        attributes["gen_ai.response.id"] = response_id
    return {"name": "gen_ai.evaluation.result", "attributes": attributes}


# ---- the demo ---------------------------------------------------------------------------------------------

# Fictional labels for 12 agent replies: a human reviewer and the LLM judge, each saying pass or fail.
HUMAN_LABELS = ["pass", "pass", "fail", "pass", "fail", "pass", "pass", "fail", "pass", "fail", "pass", "pass"]
JUDGE_LABELS = ["pass", "pass", "fail", "pass", "pass", "pass", "pass", "fail", "pass", "fail", "fail", "pass"]


def pass_hat_scores(tasks: Sequence[Task], results: dict[str, list[Trial]], k: int) -> list[float]:
    return [pass_hat_k(len(results[t.id]), sum(r.passed for r in results[t.id]), k) for t in tasks]


def run_demo(tasks: Sequence[Task], trials: int, k: int, target: float, seed: int, kappa_min: float) -> int:
    if len(tasks) < 2:
        print("Need at least 2 tasks for a standard error.", file=sys.stderr)
        return 2
    scenarios = sorted({t.scenario for t in tasks})
    print("== 1. Tasks ==")
    print(f"{len(tasks)} tasks in {len(scenarios)} scenarios ({', '.join(scenarios)}), {trials} trials each.")

    results = evaluate(tasks, AGENT_V1, trials, seed)
    all_trials = [r for rs in results.values() for r in rs]
    says_done = sum(grade_reply(r.reply) for r in all_trials)
    passed = sum(r.passed for r in all_trials)
    caught = sum(grade_reply(r.reply) and not r.passed for r in all_trials)
    shared = evaluate([replace(t, reset="shared_sandbox") for t in tasks], AGENT_V1, trials, seed)
    shared_passed = sum(r.passed for rs in shared.values() for r in rs)
    print("\n== 2. Outcome: grade the sandbox, not the reply ==")
    print(f"{len(all_trials)} trials. The reply says it is done in {says_done}; the sandbox agrees in {passed}.")
    print(f"The outcome grader caught {caught} replies that claim a change the sandbox does not show.")
    print(f"One shared sandbox per task instead of a fresh one per trial: {shared_passed} pass instead of {passed}.")

    first = tasks[0]
    row = foundry_row(first, results[first.id][0])
    print("\n== 3. Process: Microsoft Foundry Tool Use Quality (preview) ==")
    print("Testing criterion:", json.dumps(tool_use_quality_criterion(DEFAULT_JUDGE_DEPLOYMENT)))
    print(
        f"Dataset row for {first.id}, trial 1: {len(row['query'])} query messages, {len(row['response'])} response "
        f"messages, {len(row['tool_definitions'])} tool definitions. Built here, never sent: no cloud call."
    )

    question = "Did the agent book the flight it was asked for?"
    answers = {"reply_good": 0.9, "reply_weak": 0.4}
    kappa = cohen_kappa(JUDGE_LABELS, HUMAN_LABELS)
    print("\n== 4. Judge ==")
    print(f"Swap test, position-biased judge: {swap_test(first_answer_judge, question, *answers)}")
    print(f"Swap test, quality judge: {swap_test(quality_judge(answers), question, *answers)} (the better reply)")
    verdict = "calibrated" if kappa >= kappa_min else "not calibrated: fix the judge prompt before trusting it"
    print(f"Cohen's kappa, judge vs human on {len(HUMAN_LABELS)} replies: {kappa:.2f} (minimum {kappa_min:.2f}): {verdict}")

    print(f"\n== 5. pass@k and pass^k (n={trials} trials, k={k}) ==")
    print(f"{'task':<5} {'scenario':<14} {'c/n':>5} {'pass@' + str(k):>7} {'pass^' + str(k):>7}")
    at_scores = []
    for task in tasks:
        c = sum(r.passed for r in results[task.id])
        at_scores.append(pass_at_k(trials, c, k))
        print(f"{task.id:<5} {task.scenario:<14} {f'{c}/{trials}':>5} {at_scores[-1]:>7.2f} {pass_hat_k(trials, c, k):>7.2f}")
    hat_scores = pass_hat_scores(tasks, results, k)
    print(f"{'mean':<5} {'':<14} {'':>5} {_mean(at_scores):>7.2f} {_mean(hat_scores):>7.2f}")

    mean = _mean(hat_scores)
    se_plain = se_clt(hat_scores)
    se_clus = se_clustered(hat_scores, [t.scenario for t in tasks])
    low = lower_bound(mean, se_clus)
    code = gate(low, target)
    v2_scores = pass_hat_scores(tasks, evaluate(tasks, AGENT_V2, trials, seed), k)
    print("\n== 6. Gate ==")
    print(f"Mean pass^{k} {mean:.2f} over {len(tasks)} tasks. SE plain {se_plain:.3f}, clustered by scenario {se_clus:.3f}.")
    print(f"95% lower bound {low:.3f}, target {target:.2f}: {'pass' if code == 0 else 'fail'}, exit code {code}.")
    print(
        f"Agent v2 vs v1 on the same tasks: {_mean(v2_scores) - mean:+.2f} pass^{k}; "
        f"SE paired {se_paired(v2_scores, hat_scores):.3f}, unpaired {se_unpaired(v2_scores, hat_scores):.3f}."
    )
    event = evaluation_event(f"pass^{k}", round(mean, 4), "pass" if code == 0 else "fail", f"95% lower bound {low:.3f}")
    print("OpenTelemetry event:", json.dumps(event))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="An agent eval harness in six blocks, offline demo.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per task (n), at least 1")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k for pass@k and pass^k, from 1 to --trials")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="gate target for mean pass^k, 0 to 1")
    parser.add_argument("--kappa-min", type=float, default=DEFAULT_KAPPA_MIN, help="judge calibration minimum")
    parser.add_argument("--seed", type=int, default=7, help="seed for the fake agent")
    parser.add_argument("--tasks", type=Path, default=TASKS_FILE, help="task file, one JSON object per line")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if not 1 <= args.k <= args.trials:
        parser.error("--k must be from 1 to --trials")
    if not 0 <= args.target <= 1:
        parser.error("--target must be from 0 to 1")
    if not -1 <= args.kappa_min <= 1:
        parser.error("--kappa-min must be from -1 to 1")
    try:
        tasks = read_tasks(args.tasks)
    except TaskFileError as exc:
        print(f"Task file error: {exc}", file=sys.stderr)
        return 2
    return run_demo(tasks, args.trials, args.k, args.target, args.seed, args.kappa_min)


if __name__ == "__main__":
    sys.exit(main())
