"""Offline tests: fake clients for the three clouds. No network, no cloud, no credentials."""

import re
import socket
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import one_prompt_three_clouds as m

README = Path(__file__).resolve().parent.parent / "README.md"

ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.invalid",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "my-chat-deployment",
    "AWS_REGION": "eu-west-1",
    "BEDROCK_MODEL_ID": "example.model-v1:0",
    "GOOGLE_CLOUD_PROJECT": "example-project",
    "GOOGLE_CLOUD_LOCATION": "example-location",
    "GEMINI_MODEL": "example-gemini",
}
CFG = {
    "azure": {"endpoint": "https://example.invalid", "deployment": "my-chat-deployment"},
    "aws": {"region": "eu-west-1", "model": "example.model-v1:0"},
    "gcp": {"project": "example-project", "location": "example-location", "model": "example-gemini"},
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


# ---------- fakes: they record what the adapter sent and return a canned reply ----------

class FakeAzure:
    def __init__(self, response=None, error=None):
        self.calls, self.response, self.error = [], response or SimpleNamespace(output_text="azure reply"), error
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeAws:
    def __init__(self, response=None, error=None):
        self.calls, self.error = [], error
        self.response = response or {"output": {"message": {"role": "assistant", "content": [{"text": "aws reply"}]}}}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeGcp:
    def __init__(self, response=None, error=None):
        self.calls, self.error = [], error
        self.response = response or SimpleNamespace(text="gcp reply")
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def factories(azure=None, aws=None, gcp=None):
    """Client factories that hand out the fakes and record the settings they were built with."""
    built = {}

    def make(cloud, fake):
        def factory(cfg):
            built[cloud] = cfg
            return fake

        return factory

    fakes = {"azure": azure or FakeAzure(), "aws": aws or FakeAws(), "gcp": gcp or FakeGcp()}
    return {cloud: make(cloud, fake) for cloud, fake in fakes.items()}, built


def never_called(cfg):
    raise AssertionError("a client was created")


NO_CLIENTS = {cloud: never_called for cloud in ("azure", "aws", "gcp")}


# ---------- request shape: what each adapter sends ----------

def test_azure_sends_the_prompt_to_the_responses_api_with_the_deployment_name():
    client = FakeAzure()
    m.ask_azure(client, CFG["azure"])
    assert client.calls == [{"model": "my-chat-deployment", "input": m.PROMPT}]


def test_aws_sends_one_user_message_to_converse():
    client = FakeAws()
    m.ask_aws(client, CFG["aws"])
    assert client.calls == [
        {"modelId": "example.model-v1:0", "messages": [{"role": "user", "content": [{"text": m.PROMPT}]}]}
    ]


def test_gcp_sends_the_prompt_to_generate_content():
    client = FakeGcp()
    m.ask_gcp(client, CFG["gcp"])
    assert client.calls == [{"model": "example-gemini", "contents": m.PROMPT}]


def test_all_three_clouds_get_the_same_prompt(capsys):
    azure, aws, gcp = FakeAzure(), FakeAws(), FakeGcp()
    fakes, _ = factories(azure, aws, gcp)
    assert m.main([], env=ENV, client_factories=fakes) == 0
    sent = [azure.calls[0]["input"], aws.calls[0]["messages"][0]["content"][0]["text"], gcp.calls[0]["contents"]]
    assert sent == [m.PROMPT] * 3


def test_each_factory_gets_only_its_own_settings():
    fakes, built = factories()
    m.main([], env=ENV, client_factories=fakes)
    assert built == CFG


# ---------- reply text: extraction, including an empty reply ----------

def test_azure_reply_text():
    assert m.ask_azure(FakeAzure(), CFG["azure"]) == "azure reply"


def test_aws_reply_text_joins_text_blocks_and_skips_other_blocks():
    content = [{"reasoningContent": {"reasoningText": {"text": "thinking"}}}, {"text": "Hello, "}, {"text": "world"}]
    client = FakeAws({"output": {"message": {"role": "assistant", "content": content}}})
    assert m.ask_aws(client, CFG["aws"]) == "Hello, world"


def test_gcp_reply_text():
    assert m.ask_gcp(FakeGcp(), CFG["gcp"]) == "gcp reply"


def test_empty_replies_come_back_empty():
    assert m.ask_azure(FakeAzure(SimpleNamespace(output_text="")), CFG["azure"]) == ""
    assert m.ask_aws(FakeAws({"output": {"message": {"role": "assistant", "content": []}}}), CFG["aws"]) == ""
    assert m.ask_gcp(FakeGcp(SimpleNamespace(text=None)), CFG["gcp"]) == ""  # the SDK returns None when nothing came back


def test_an_empty_reply_is_printed_as_empty_not_as_a_crash(capsys):
    fakes, _ = factories(gcp=FakeGcp(SimpleNamespace(text=None)))
    assert m.main(["--cloud", "gcp"], env=ENV, client_factories=fakes) == 0
    assert "(empty reply)" in capsys.readouterr().out


# ---------- regression: each adapter reads the right reply field and ignores look-alike fields ----------

def test_regression_azure_reads_output_text_not_the_chat_completions_path():
    response = SimpleNamespace(
        output_text="RIGHT",
        choices=[SimpleNamespace(message=SimpleNamespace(content="WRONG"))],  # Chat Completions shape
        output=[SimpleNamespace(content=[SimpleNamespace(text="WRONG")])],    # raw Responses items
    )
    assert m.ask_azure(FakeAzure(response), CFG["azure"]) == "RIGHT"


def test_regression_aws_reads_output_message_content_text():
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "RIGHT"}]}},
        "stopReason": "WRONG",
        "choices": [{"message": {"content": "WRONG"}}],  # Chat Completions shape
    }
    assert m.ask_aws(FakeAws(response), CFG["aws"]) == "RIGHT"


def test_regression_gcp_reads_text_not_a_candidate_part():
    response = SimpleNamespace(
        text="RIGHT",
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="WRONG")]))],
    )
    assert m.ask_gcp(FakeGcp(response), CFG["gcp"]) == "RIGHT"


def test_aws_adapter_matches_the_real_converse_request_and_response_shape(monkeypatch):
    """botocore's own service model checks the request and parses the response. No credentials, no network."""
    import os

    import boto3
    from botocore.stub import Stubber

    for name in [n for n in os.environ if n.startswith("AWS_")]:  # keep the developer's own credentials out
        monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")  # no instance-metadata lookup
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    client = boto3.client("bedrock-runtime", region_name="eu-west-1")
    reply = {
        "output": {"message": {"role": "assistant", "content": [{"text": "from the service model"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }
    request = {"modelId": "example.model-v1:0", "messages": [{"role": "user", "content": [{"text": m.PROMPT}]}]}
    with Stubber(client) as stubber:
        stubber.add_response("converse", reply, request)
        assert m.ask_aws(client, CFG["aws"]) == "from the service model"
        stubber.assert_no_pending_responses()


# ---------- the real client factories: wiring only, nothing is sent ----------

def test_azure_client_uses_the_v1_base_url_and_an_entra_token_provider(monkeypatch):
    import azure.identity

    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda: object())  # no sign-in attempted
    client = m.azure_client({"endpoint": "https://example.invalid/", "deployment": "d"})
    assert str(client.base_url) == "https://example.invalid/openai/v1/"


def test_azure_client_passes_an_entra_token_provider_for_the_foundry_scope_as_the_api_key(monkeypatch):
    import azure.identity
    import openai

    credential, provider, seen = object(), (lambda: "token"), {}
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda: credential)
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider",
                        lambda cred, scope: seen.update(cred=cred, scope=scope) or provider)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: seen.update(kwargs) or SimpleNamespace(**kwargs))
    m.azure_client({"endpoint": "https://example.invalid", "deployment": "d"})
    assert seen["cred"] is credential
    assert seen["scope"] == "https://ai.azure.com/.default"
    assert seen["api_key"] is provider  # a token provider, never a key string


def test_aws_client_is_bedrock_runtime_in_the_configured_region(monkeypatch):
    import boto3

    seen = []
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: seen.append((args, kwargs)))
    m.aws_client(CFG["aws"])
    assert seen == [(("bedrock-runtime",), {"region_name": "eu-west-1"})]


def test_gcp_client_selects_the_enterprise_agent_platform_with_no_api_key(monkeypatch):
    import inspect

    from google import genai

    assert "enterprise" in inspect.signature(genai.Client.__init__).parameters  # the installed SDK still has the flag
    seen = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: seen.append(kwargs))
    m.gcp_client(CFG["gcp"])
    assert seen == [{"enterprise": True, "project": "example-project", "location": "example-location"}]


# ---------- configuration: exit 2, skipping, per-cloud errors ----------

@pytest.mark.parametrize("cloud", ["azure", "aws", "gcp"])
def test_a_named_cloud_with_missing_settings_exits_2(cloud, capsys):
    code = m.main(["--cloud", cloud], env={}, client_factories=NO_CLIENTS)
    out = capsys.readouterr().out
    assert code == 2
    assert all(var in out for var in m.SETTINGS[cloud].values())


def test_a_named_cloud_with_one_missing_setting_names_only_that_one(capsys):
    env = {k: v for k, v in ENV.items() if k != "BEDROCK_MODEL_ID"}
    assert m.main(["--cloud", "aws"], env=env, client_factories=NO_CLIENTS) == 2
    out = capsys.readouterr().out
    assert "BEDROCK_MODEL_ID" in out and "AWS_REGION" not in out


def test_a_blank_setting_counts_as_missing(capsys):
    assert m.main(["--cloud", "azure"], env={**ENV, "AZURE_OPENAI_ENDPOINT": "  "}, client_factories=NO_CLIENTS) == 2


def test_a_setting_left_as_its_env_example_placeholder_counts_as_missing(capsys):
    example = Path(__file__).resolve().parent.parent / ".env.example"
    env = dict(line.split("=", 1) for line in example.read_text().splitlines() if line and not line.startswith("#"))
    env.update({"AWS_REGION": "eu-west-1", "BEDROCK_MODEL_ID": "example.model-v1:0"})  # only AWS filled in
    aws = FakeAws()
    fakes, built = factories(aws=aws)
    assert m.main([], env=env, client_factories=fakes) == 0
    out = capsys.readouterr().out
    assert "Skipping azure" in out and "Skipping gcp" in out
    assert list(built) == ["aws"]


def test_a_named_cloud_ignores_the_other_clouds_settings(capsys):
    azure, aws = FakeAzure(), FakeAws()
    fakes, _ = factories(azure, aws)
    env = {k: v for k, v in ENV.items() if k.startswith("AWS") or k.startswith("BEDROCK")}
    assert m.main(["--cloud", "aws"], env=env, client_factories=fakes) == 0
    assert aws.calls and not azure.calls
    assert "aws reply" in capsys.readouterr().out


def test_all_skips_unconfigured_clouds_and_runs_the_configured_ones(capsys):
    aws = FakeAws()
    fakes, built = factories(aws=aws)
    env = {k: v for k, v in ENV.items() if k.startswith("AWS") or k.startswith("BEDROCK")}
    assert m.main([], env=env, client_factories=fakes) == 0
    out = capsys.readouterr().out
    assert "Skipping azure: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT" in out
    assert "Skipping gcp: set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GEMINI_MODEL" in out
    assert "aws reply" in out
    assert list(built) == ["aws"]


def test_all_with_nothing_configured_exits_2(capsys):
    assert m.main(["--cloud", "all"], env={}, client_factories=NO_CLIENTS) == 2
    assert "No cloud is configured" in capsys.readouterr().out


def test_an_adapter_error_is_reported_for_that_cloud_and_the_others_still_run(capsys):
    fakes, _ = factories(aws=FakeAws(error=RuntimeError("AccessDeniedException: not authorized\nrequest id 123")))
    assert m.main([], env=ENV, client_factories=fakes) == 1
    out = capsys.readouterr().out
    assert "ERROR RuntimeError: AccessDeniedException: not authorized" in out
    assert "request id 123" not in out  # one line per error
    assert "azure reply" in out and "gcp reply" in out


def test_a_client_that_cannot_be_built_is_reported_for_that_cloud_only(capsys):
    def no_credentials(cfg):
        raise RuntimeError("no Application Default Credentials found")

    fakes, _ = factories()
    fakes["gcp"] = no_credentials
    assert m.main([], env=ENV, client_factories=fakes) == 1
    out = capsys.readouterr().out
    assert "ERROR RuntimeError: no Application Default Credentials found" in out
    assert "azure reply" in out and "aws reply" in out


def test_an_invalid_cloud_exits_2():
    with pytest.raises(SystemExit) as exc:
        m.main(["--cloud", "oracle"], env=ENV, client_factories=NO_CLIENTS)
    assert exc.value.code == 2


# ---------- --map: the 13-layer table ----------

def test_the_stack_data_has_13_layers_with_three_complete_entries_and_an_https_source():
    assert len(m.STACK) == 13
    assert [layer.split()[0] for layer, *_ in m.STACK] == [str(n) for n in range(1, 14)]
    for layer, *entries in m.STACK:
        assert len(entries) == 3, layer
        for entry in entries:
            assert entry.name.strip() and entry.status.strip(), layer
            parsed = urlparse(entry.url)
            assert parsed.scheme == "https" and parsed.netloc and " " not in entry.url, (layer, entry.url)


def test_map_prints_every_layer_with_three_non_empty_entries_and_makes_no_call(capsys):
    assert m.main(["--map"], env={}, client_factories=NO_CLIENTS) == 0
    blocks = capsys.readouterr().out.strip().split("\n\n")
    assert len(blocks) == 13
    for block, (layer, *_) in zip(blocks, m.STACK):
        lines = block.splitlines()
        assert lines[0] == layer
        body = "\n".join(lines[1:])
        for label in ("Azure", "AWS", "Google Cloud"):
            found = re.findall(rf"^  {label} +(\S.*)$", body, flags=re.M)
            assert len(found) == 1 and found[0].strip(), (layer, label)


def test_map_prints_names_and_statuses_from_the_data(capsys):
    m.print_map()
    flat = " ".join(capsys.readouterr().out.split())
    for _, *entries in m.STACK:
        for entry in entries:
            assert " ".join(f"{entry.name} | {entry.status}".split()) in flat
            assert entry.url in flat  # the README promises a source link for every entry


def test_the_readme_table_matches_the_stack_data():
    text = README.read_text(encoding="utf-8")
    for layer, *entries in m.STACK:
        assert layer in text, layer
        for entry in entries:
            assert f"[{entry.name}]({entry.url})" in text, entry.name
            assert entry.status in text, entry.status
