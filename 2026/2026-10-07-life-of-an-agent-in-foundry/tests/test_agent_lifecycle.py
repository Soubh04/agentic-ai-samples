"""Offline tests: a fake project client and fake OpenAI client. No network, no Azure, no env vars."""

import json
from types import SimpleNamespace as NS

import pytest
from azure.ai.projects.models import AgentEndpointConfig, FunctionTool, PromptAgentDefinition

import agent_lifecycle as al

ANSWERS = dict(al.QUESTIONS)  # question -> correct ticket id


class FakeOpenAI:
    """Calls get_ticket_status for each question; `wrong` makes it look up the wrong id."""

    def __init__(self, wrong=False):
        self.wrong, self.calls, self.responses = wrong, [], self
        self.last_question = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def create(self, input=None, previous_response_id=None, extra_body=None):
        self.calls.append({"input": input, "previous_response_id": previous_response_id, "extra_body": extra_body})
        if isinstance(input, str):
            self.last_question = input
            ticket = "INC-0000" if self.wrong else ANSWERS[input]
            call = NS(type="function_call", name="get_ticket_status", call_id="c1",
                      arguments=json.dumps({"ticket_id": ticket}))
            return NS(id="r1", output=[call])
        return NS(id="r2", output=[NS(type="message")])


class FakeAgents:
    def __init__(self):
        self.created, self.updates = [], []

    def create_version(self, agent_name, definition):
        self.created.append((agent_name, definition))
        return NS(id=f"{agent_name}:1", name=agent_name, version="1")

    def get(self, agent_name):
        return NS(agent_endpoint=None)

    def update_details(self, agent_name, agent_endpoint):
        self.updates.append((agent_name, agent_endpoint))


class FakeProject:
    def __init__(self, wrong=False):
        self.agents, self.openai = FakeAgents(), FakeOpenAI(wrong)

    def get_openai_client(self, agent_name=None):
        assert agent_name is None  # project-level client; each call names the agent version
        return self.openai


def routed_version(update):
    return update[1].version_selector.version_selection_rules[0].agent_version


def test_golden_path_scores_1_and_routes(capsys):
    project = FakeProject()
    assert al.run(project, "my-model") == 0
    name, definition = project.agents.created[0]
    assert isinstance(definition, PromptAgentDefinition) and definition.model == "my-model"
    assert isinstance(definition.tools[0], FunctionTool) and definition.tools[0].name == "get_ticket_status"
    assert all(isinstance(cfg, AgentEndpointConfig) for _, cfg in project.agents.updates)
    last = project.agents.updates[-1]
    rule = last[1].version_selector.version_selection_rules[0]
    assert (last[0], rule.agent_version, rule.traffic_percentage) == ("helpdesk-agent", "1", 100)
    assert "1.00" in capsys.readouterr().out
    # the tool result went back with previous_response_id
    followups = [c for c in project.openai.calls if c["previous_response_id"]]
    assert len(followups) == len(al.QUESTIONS)
    assert followups[0]["input"][0]["type"] == "function_call_output"
    # every call went to the new version by name, before any routing
    refs = {(c["extra_body"]["agent_reference"]["name"], c["extra_body"]["agent_reference"]["version"])
            for c in project.openai.calls}
    assert refs == {("helpdesk-agent", "1")}


def test_below_bar_is_not_routed(capsys):
    project = FakeProject(wrong=True)
    assert al.run(project, "my-model") == 1
    assert "below the bar, not routed" in capsys.readouterr().out
    assert project.agents.updates == []  # regression: a failing version must never receive traffic


def test_missing_env_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_MODEL_NAME", raising=False)
    assert al.main([]) == 2
    err = capsys.readouterr().err
    assert "FOUNDRY_PROJECT_ENDPOINT" in err and "az login" in err and ".env.example" in err


def test_rollback_routes_to_given_version():
    project = FakeProject()
    assert al.run(project, "my-model", rollback_to="3") == 0
    assert [routed_version(u) for u in project.agents.updates] == ["3"]
    assert project.agents.created == []


def test_rollback_without_version_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://example.invalid/api/projects/p")
    monkeypatch.setenv("FOUNDRY_MODEL_NAME", "my-model")
    assert al.main(["--rollback-to"]) == 2
    assert "Usage" in capsys.readouterr().err


def test_unknown_ticket_is_not_found():
    assert al.get_ticket_status("INC-9999") == {"error": "ticket INC-9999 not found"}
    assert al.get_ticket_status("INC-1042")["status"] == "in progress"


def test_no_secrets_in_env_example():
    text = open(".env.example").read()
    assert "<your-" in text and "api-key" not in text.lower()
