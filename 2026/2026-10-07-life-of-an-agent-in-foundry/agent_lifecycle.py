"""Life of an AI agent in Microsoft Foundry: Build, Test, Evaluate, Route, Roll back.

Bring your own Azure subscription and Foundry project. Sign in with `az login`
(Entra ID, no API keys). Set FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL_NAME (see .env.example).

    uv run --env-file .env python agent_lifecycle.py
    uv run --env-file .env python agent_lifecycle.py --rollback-to 1

Exit codes: 0 ok, 1 below the quality bar (not routed), 2 missing configuration.
"""

import json
import os
import sys

from azure.ai.projects.models import (
    AgentEndpointConfig,
    FixedRatioVersionSelectionRule,
    FunctionTool,
    PromptAgentDefinition,
    ProtocolConfiguration,
    ResponsesProtocolConfiguration,
    VersionSelector,
)

AGENT_NAME = "helpdesk-agent"
BAR = 0.75
TICKETS = {  # fictional data
    "INC-1042": {"status": "in progress", "assignee": "Mira"},
    "INC-1050": {"status": "open", "assignee": "Jonas"},
    "INC-1033": {"status": "resolved", "assignee": "Noor"},
}
QUESTIONS = [  # (question, ticket id the agent should look up)
    ("What is the status of INC-1042?", "INC-1042"),
    ("Who is the assignee for INC-1050?", "INC-1050"),
    ("Is INC-1033 resolved?", "INC-1033"),
    ("Any news on INC-9999?", "INC-9999"),
]
TOOL = FunctionTool(
    name="get_ticket_status",
    description="Look up the status and assignee of an incident ticket by its id, for example INC-1042.",
    parameters={
        "type": "object",
        "properties": {"ticket_id": {"type": "string", "description": "Ticket id such as INC-1042."}},
        "required": ["ticket_id"],
        "additionalProperties": False,
    },
    strict=True,
)


def get_ticket_status(ticket_id):
    return TICKETS.get(ticket_id, {"error": f"ticket {ticket_id} not found"})


def route(project_client, version):
    """Send 100% of the agent's stable endpoint to one version."""
    config = AgentEndpointConfig(
        version_selector=VersionSelector(
            version_selection_rules=[FixedRatioVersionSelectionRule(agent_version=version, traffic_percentage=100)]
        ),
        protocol_configuration=ProtocolConfiguration(responses=ResponsesProtocolConfiguration()),
    )
    project_client.agents.update_details(agent_name=AGENT_NAME, agent_endpoint=config)


def ask(openai_client, question, version):
    """Ask one version of the agent a question; answer function_call items locally.
    Returns the ticket ids the agent looked up."""
    agent = {"agent_reference": {"type": "agent_reference", "name": AGENT_NAME, "version": version}}
    response = openai_client.responses.create(input=question, extra_body=agent)
    looked_up = []
    for _ in range(5):
        outputs = []
        for item in response.output:
            if item.type == "function_call" and item.name == "get_ticket_status":
                args = json.loads(item.arguments)
                looked_up.append(args.get("ticket_id"))
                outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(get_ticket_status(**args)),
                })
        if not outputs:
            break
        response = openai_client.responses.create(input=outputs, previous_response_id=response.id, extra_body=agent)
    return looked_up


def score(project_client, version):
    """Share of questions where this version called get_ticket_status with the right ticket id."""
    hits = 0
    with project_client.get_openai_client() as openai_client:
        for question, ticket_id in QUESTIONS:
            looked_up = ask(openai_client, question, version)
            ok = looked_up == [ticket_id]
            hits += ok
            print(f"   {'ok  ' if ok else 'MISS'} {question}  ->  {looked_up}")
    return hits / len(QUESTIONS)


def run(project_client, model, rollback_to=None):
    if rollback_to:
        print(f"Roll back: routing 100% to {AGENT_NAME}:{rollback_to}")
        route(project_client, rollback_to)
        return 0
    definition = PromptAgentDefinition(
        model=model, instructions="You are an IT helpdesk assistant. Use the tool to look up tickets.", tools=[TOOL]
    )
    version = project_client.agents.create_version(agent_name=AGENT_NAME, definition=definition)
    print(f"1 Build     saved {AGENT_NAME}:{version.version} (versions are immutable)")
    print(f"2 Test      asking {AGENT_NAME}:{version.version} {len(QUESTIONS)} questions (no live traffic yet)")
    result = score(project_client, version.version)
    print(f"3 Evaluate  tool-call accuracy {result:.2f} (bar {BAR:.2f})")
    print("            in Foundry use the built-in evaluator builtin.tool_call_accuracy (LLM-judged)")
    if result >= BAR:
        route(project_client, version.version)
        print(f"4 Route     100% of traffic -> {AGENT_NAME}:{version.version}")
        return 0
    print("4 Route     below the bar, not routed (traffic stays on the current version)")
    return 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    endpoint, model = os.environ.get("FOUNDRY_PROJECT_ENDPOINT"), os.environ.get("FOUNDRY_MODEL_NAME")
    if not endpoint or not model:
        print(
            "Bring your own Foundry project: set FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL_NAME "
            "(see .env.example), then run `az login`.",
            file=sys.stderr,
        )
        return 2
    rollback_to = None
    if "--rollback-to" in argv:
        i = argv.index("--rollback-to") + 1
        if i >= len(argv):
            print("Usage: agent_lifecycle.py [--rollback-to VERSION]", file=sys.stderr)
            return 2
        rollback_to = argv[i]
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=endpoint, credential=credential) as client:
        return run(client, model, rollback_to)


if __name__ == "__main__":
    sys.exit(main())
