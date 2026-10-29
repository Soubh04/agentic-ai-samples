# Evaluating AI agents: what to measure before you ship

A small Python script that runs the six blocks of an agent eval harness from the LinkedIn post "Evaluating AI agents:
what to measure before you ship". It loads a task file, runs a seeded fake travel agent several times per task, grades
the sandbox instead of the agent's reply, builds a Microsoft Foundry Tool Use Quality configuration, tests an LLM judge
for position bias and agreement with human labels, estimates pass@k and pass^k, and turns a 95% lower bound into a CI
exit code.

LinkedIn post: https://www.linkedin.com/posts/soubh_agenticai-aiagents-llmevaluation-share-7521280093390909442-FxeD/

> **This sample calls no cloud service and no model.** The agent is a seeded fake that does the job, claims it without
> doing it, or hits a timeout. The script builds the Foundry testing criterion and a dataset row in the documented
> shape but never sends them: it was not run against a live Microsoft Foundry project by the author. Tool Use Quality
> is in preview, and the OpenTelemetry GenAI semantic conventions are in Development.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.10 or later
- Nothing else. This sample calls no cloud service, needs no credentials, and uses only the standard library.

## Setup and run

```bash
uv run python agent_evals.py                        # exits 1 on purpose: see below
uv run python agent_evals.py --k 1                  # pass^1: the same 8 trials per task, one success needed
uv run python agent_evals.py --trials 1 --k 1       # one run per task and one average: the post's mistake
uv run python agent_evals.py --trials 16 --k 4      # more trials per task
```

| Option | Default | Meaning |
|---|---|---|
| `--trials` | 8 | Trials per task (n), at least 1 |
| `--k` | 4 | k for pass@k and pass^k, from 1 to `--trials` |
| `--target` | 0.25 | The gate: the 95% lower bound of mean pass^k must reach this value, from 0 to 1 |
| `--kappa-min` | 0.6 | The judge counts as calibrated when Cohen's kappa against human labels reaches this value, from -1 to 1 |
| `--seed` | 7 | Seed for the fake agent |
| `--tasks` | `tasks.jsonl` | Task file, one JSON object per line |

Exit codes: 0 when the gate passes, 1 when it fails, 2 for an invalid argument or task file. The target and the kappa
minimum are the author's demo values, not standards: set them from what your agent must deliver.

A task's `scenario` sets the fake agent's chance of doing the job (`AGENT_V1` and `AGENT_V2` in the script). In your own
task file, a scenario the script does not name, or a task without a `scenario` field, gets a chance of 0.5.

What the default run prints:

```text
== 1. Tasks ==
8 tasks in 3 scenarios (cancel, domestic, international), 8 trials each.

== 2. Outcome: grade the sandbox, not the reply ==
64 trials. The reply says it is done in 51; the sandbox agrees in 45.
The outcome grader caught 6 replies that claim a change the sandbox does not show.
One shared sandbox per task instead of a fresh one per trial: 59 pass instead of 45.

== 3. Process: Microsoft Foundry Tool Use Quality (preview) ==
Testing criterion: {"type": "azure_ai_evaluator", "name": "tool_use_quality", "evaluator_name": "builtin.tool_use_quality", "initialization_parameters": {"deployment_name": "gpt-5.6-luna", "evaluation_level": "turn"}, "data_mapping": {"query": "{{item.query}}", "response": "{{item.response}}", "tool_definitions": "{{item.tool_definitions}}"}}
Dataset row for t01, trial 1: 2 query messages, 3 response messages, 3 tool definitions. Built here, never sent: no cloud call.

== 4. Judge ==
Swap test, position-biased judge: tie
Swap test, quality judge: x (the better reply)
Cohen's kappa, judge vs human on 12 replies: 0.63 (minimum 0.60): calibrated

== 5. pass@k and pass^k (n=8 trials, k=4) ==
task  scenario         c/n  pass@4  pass^4
t01   domestic         7/8    1.00    0.50
t02   domestic         8/8    1.00    1.00
t03   domestic         6/8    1.00    0.21
t04   international    6/8    1.00    0.21
t05   international    4/8    0.99    0.01
t06   cancel           5/8    1.00    0.07
t07   international    3/8    0.93    0.00
t08   cancel           6/8    1.00    0.21
mean                          0.99    0.28

== 6. Gate ==
Mean pass^4 0.28 over 8 tasks. SE plain 0.117, clustered by scenario 0.144.
95% lower bound -0.003, target 0.25: fail, exit code 1.
Agent v2 vs v1 on the same tasks: +0.11 pass^4; SE paired 0.066, unpaired 0.186.
OpenTelemetry event: {"name": "gen_ai.evaluation.result", "attributes": {"gen_ai.evaluation.name": "pass^4", "gen_ai.evaluation.score.value": 0.2786, "gen_ai.evaluation.score.label": "fail", "gen_ai.evaluation.explanation": "95% lower bound -0.003"}}
```

The default run fails its gate on purpose. Mean pass^4 is 0.28, above the 0.25 target, but 8 tasks in 3 scenarios
give a clustered standard error of 0.144, and the 95% lower bound sits just below zero. The plain standard error would
put it at 0.049, still far below the target. A gate on the mean alone would have shown a green tick. The same agent at
`--k 1` passes, because one success per task is a much easier bar than four in a row:

```text
== 6. Gate ==
Mean pass^1 0.70 over 8 tasks. SE plain 0.071, clustered by scenario 0.092.
95% lower bound 0.523, target 0.25: pass, exit code 0.
```

One run per task and one average, the mistake the post describes, also gets a green tick for this agent:

```text
== 6. Gate ==
Mean pass^1 0.50 over 8 tasks. SE plain 0.189, clustered by scenario 0.111.
95% lower bound 0.283, target 0.25: pass, exit code 0.
```

## Post to code

| Post | What the script does | Code |
|---|---|---|
| 1. Tasks: draw them from real failures; reset the environment per trial | One JSON line per task with a goal, the expected end state and a reset rule. Each trial gets a fresh sandbox. The demo reruns every task in one shared sandbox, and the leftover bookings lift the pass count from 45 to 59 | `tasks.jsonl`, `load_tasks()`, `new_env()`, `run_task()` |
| 2. Outcome: "Your flight is booked" is a claim | `grade()` counts the rows the agent left in the sandbox and never reads the reply. `grade_reply()` is the reply-trusting grader the post warns about | `grade()`, `grade_reply()` |
| 3. Trajectory: no single required tool sequence; Foundry process evaluators | The outcome grade passes an agent that takes a longer, still valid tool path. The script builds the Tool Use Quality testing criterion and one dataset row (query, response with tool calls and results, tool definitions) | `tool_use_quality_criterion()`, `foundry_row()`, `TOOL_DEFINITIONS` |
| 4. Judge: position bias; calibrate against human experts | A swap test judges both orders and counts a win only when both verdicts agree, so a judge that always prefers the first answer ends in a tie. Cohen's kappa measures agreement with human labels beyond chance | `swap_test()`, `cohen_kappa()` |
| 5. pass@k or pass^k | The unbiased estimators from n trials with c passes, per task, with the mean across tasks | `pass_at_k()`, `pass_hat_k()` |
| 6. Gate: standard error, clustered; paired differences; OpenTelemetry | Standard errors plain and clustered by scenario, a paired comparison of two agent versions on the same tasks, the 95% lower bound against the target as the exit code, and the `gen_ai.evaluation.result` event attributes | `se_clt()`, `se_clustered()`, `se_paired()`, `lower_bound()`, `gate()`, `evaluation_event()` |
| The mistake: one run per task, one average, a green tick | `--trials 1 --k 1` runs each task once and the gate passes. The default run measures the same agent with 8 trials at pass^4, and the gate fails | `run_demo()`, `main()` |

The image's six code blocks map to the script like this:

| Image block | In the script |
|---|---|
| Write the task file | Task `t07` in `tasks.jsonl` |
| Grade the end state | `grade()`, which checks every table named in `expect`, not only `bookings` |
| Score the tool calls | `tool_use_quality_criterion()`. The image shows part of the dictionary; the script builds all of it |
| Test the judge | `swap_test()`, `cohen_kappa()` and the kappa check in `run_demo()`. The demo prints the verdict instead of stopping; in CI, fail the run |
| Estimate pass^k | `pass_at_k()` and `pass_hat_k()`. A test checks the worked example: n=8, c=6, k=4 gives pass@4 1.00 and pass^4 0.21 |
| Gate the build | `se_bernoulli()`, `lower_bound()`, `gate()` and `evaluation_event()` |

The image's gate uses `sqrt(p * (1 - p) / n)`, the standard error for pass/fail scores. A per-task pass^k is a number
between 0 and 1, not a pass or a fail, so the demo uses the general standard error of a mean (`se_clt()`, equation 1 in
Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)) and its clustered form (`se_clustered()`,
equation 4). For pass/fail scores the two give almost the same number, and a test checks that.

The Foundry configuration shape and dataset format follow
[Agent evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/agent-evaluators).
That page recommends a GPT-5.6-family model as the judge for composite evaluators, with `gpt-5.6-luna` as the
lowest-cost option. The deployment name in the script is a placeholder for your own deployment. The event attributes
follow the OpenTelemetry
[GenAI events](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)
conventions.

## What the script leaves out

- A model. The agent is a seeded fake with a fixed chance per scenario of doing the job, and the judges are stand-ins.
  A real judge needs a prompt, one dimension per judge, and an "Unknown" answer it can give.
- The Foundry call. To run the evaluator, pass the testing criterion and a dataset of rows like `foundry_row()` to an
  evaluation run in your Foundry project. See
  [Introduction to cloud evaluation with Microsoft Foundry SDK](https://learn.microsoft.com/en-us/azure/foundry/observability/how-to/cloud-evaluation).
- An OpenTelemetry exporter. The script prints the event's attributes. A real harness emits the event through an
  OpenTelemetry SDK, parented to the span of the GenAI operation it evaluates.
- Better intervals. The 95% interval is the normal approximation, which can fall below zero with few tasks, as in the
  default run. Wilson or bootstrap intervals behave better at small sample sizes.
- An interval for kappa. The demo compares one point estimate from 12 fictional labels with the minimum. With real
  labels, use many more items and put an interval on kappa as well.
- Many clusters. The demo has three scenarios, so its clustered standard error is itself a rough estimate.
- Realistic pairing. The two fake agent versions share random draws per task, which makes the paired comparison
  unusually tight. How much pairing helps in practice depends on how strongly two versions' scores move together.
- A tool-sequence evaluator. The script grades the end state and builds the Foundry configuration; it does not check
  the order of tool calls itself.

## Tests

```bash
uv run pytest -q
```

They run offline, with no model, network or credentials. They cover the task file (the shipped file, defaults, blank
lines, each invalid field, duplicate ids, an empty, missing or non-UTF-8 file), the outcome grade for book and cancel
tasks and for two tables in `expect`, a claim that fools the reply grader but not the outcome grader, an outcome grade
that accepts a longer valid tool path, score inflation from a shared sandbox, the fake agent's three outcomes and its
early stop, the Foundry criterion and dataset row shapes, the swap test and Cohen's kappa, the pass@k and pass^k
estimators with their edge cases and the image's worked example, the interval figures from the post and the image,
plain, clustered and paired standard errors checked against hand-computed values, the gate boundary, the OpenTelemetry
attribute names, the figures the default run prints, one run per task passing the gate, an uncalibrated judge, and
exit codes 0, 1 and 2.

```text
.................................................                        [100%]
49 passed in 0.09s
```

All data is fictional.
