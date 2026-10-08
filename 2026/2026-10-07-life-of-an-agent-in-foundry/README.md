# The life of an AI agent in Microsoft Foundry

A small Python script that walks one agent through its life in Microsoft Foundry:
**Build, Test, Evaluate, Route, Roll back**. It accompanies the LinkedIn post "The life of an AI agent in Microsoft Foundry".

LinkedIn post: (link added after publishing)

> **Not run against a live Foundry project by the author.** The script's SDK calls follow the official
> [azure-ai-projects samples](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-projects/samples)
> (`sample_agent_basic.py`, `sample_agent_function_tool.py`, `sample_agent_response_evaluation.py`) and are unit-tested against fakes.

## Bring your own Azure subscription and Foundry project

You need:

- an Azure subscription and a Microsoft Foundry project
- a model deployed in that project (any model that supports tool calling)
- the Azure CLI, signed in with `az login` (Foundry agents use Microsoft Entra ID sign-in, **no API keys**)
- [uv](https://docs.astral.sh/uv/) and Python 3.10+

Nothing in this repo holds credentials. Your endpoint and model name stay in your own `.env` (git-ignored).

## Setup and run

```bash
cp .env.example .env        # then edit .env: your project endpoint and model deployment name
az login
uv run --env-file .env python agent_lifecycle.py
uv run --env-file .env python agent_lifecycle.py --rollback-to 1   # one call, no redeploy
```

Without the two variables the script prints what to set and exits with code 2.
Exit codes: `0` ok, `1` below the quality bar (not routed), `2` missing configuration.

Note: running it creates a new version of an agent named `helpdesk-agent` in your project and may change
its endpoint routing. Use a throwaway project.

## Stage to code

| Post stage | What happens | Code |
|---|---|---|
| 1 Build | Save a prompt agent version with one function tool, `get_ticket_status` (fictional tickets in the file) | `PromptAgentDefinition`, `FunctionTool`, `agents.create_version` |
| 2 Test | Ask the new version 4 questions by name, before it gets any traffic; answer `function_call` items locally, send `function_call_output` back with `previous_response_id` | `get_openai_client()`, `responses.create(extra_body={"agent_reference": ...})`, `ask()` |
| 3 Evaluate | Local check: did it call the tool with the right ticket id? Score 0 to 1 | `score()`. In Foundry use the built-in [`builtin.tool_call_accuracy`](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/agent-evaluators) evaluator (LLM-judged) |
| 4 Route | Score 0.75 or more: send 100% of the endpoint to this version. Otherwise nothing changes; traffic stays on the current version | `AgentEndpointConfig`, `VersionSelector`, `FixedRatioVersionSelectionRule`, `agents.update_details` |
| Roll back | `--rollback-to N` points the endpoint at version N | `route()` |

## Tests

```bash
uv run pytest -q
```

They use a fake project client and fake OpenAI client: no network, no Azure, no credentials. They cover the golden path
(score 1.0, routed), the below-bar path (never routed), missing configuration (exit 2), rollback with and without a
version, and an unknown ticket id.

```text
.......                                                                  [100%]
7 passed in 1.03s
```

All data is fictional.
