"""One prompt, three clouds: the agent stack on Azure, AWS and Google Cloud.

Sends the same prompt to one chat model on each cloud and prints each reply. Layer 1 of the stack (the model
catalog) is the only layer every cloud lets you call with a few lines, so it is the one this script runs.
`--map` prints all 13 layers of the managed agent stack, with each cloud's official name and status.

  Azure         the `openai` SDK on the Azure OpenAI v1 endpoint, Responses API, Microsoft Entra ID sign-in
  AWS           boto3 `bedrock-runtime`, Converse API, the standard AWS credential chain
  Google Cloud  `google-genai` on the Gemini Enterprise Agent Platform, Application Default Credentials

No API keys anywhere. Set the variables in .env.example, then sign in on each cloud (see the README).

    uv run --env-file .env python one_prompt_three_clouds.py                 # every configured cloud
    uv run --env-file .env python one_prompt_three_clouds.py --cloud aws     # one cloud
    uv run python one_prompt_three_clouds.py --map                           # the 13-layer table, no network

Exit codes: 0 ok, 1 at least one cloud call failed, 2 missing configuration or invalid argument.
"""

import argparse
import os
import sys
import textwrap
from collections import namedtuple

PROMPT = "In one sentence, what does an agent runtime add on top of a plain model API?"

# Settings come from environment variables only. cloud -> {setting: variable name}
SETTINGS = {
    "azure": {"endpoint": "AZURE_OPENAI_ENDPOINT", "deployment": "AZURE_OPENAI_CHAT_DEPLOYMENT"},
    "aws": {"region": "AWS_REGION", "model": "BEDROCK_MODEL_ID"},
    "gcp": {"project": "GOOGLE_CLOUD_PROJECT", "location": "GOOGLE_CLOUD_LOCATION", "model": "GEMINI_MODEL"},
}
TITLES = {
    "azure": "Azure: Microsoft Foundry, Responses API",
    "aws": "AWS: Amazon Bedrock, Converse API",
    "gcp": "Google Cloud: Gemini Enterprise Agent Platform, generate_content",
}


# ---------- one short function per cloud: client factory, then the call ----------

def azure_client(cfg):
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import OpenAI

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
    return OpenAI(base_url=f"{cfg['endpoint'].rstrip('/')}/openai/v1/", api_key=token_provider)


def ask_azure(client, cfg):
    response = client.responses.create(model=cfg["deployment"], input=PROMPT)  # model = your deployment name
    return response.output_text


def aws_client(cfg):
    import boto3

    return boto3.client("bedrock-runtime", region_name=cfg["region"])  # credentials: the standard AWS chain


def ask_aws(client, cfg):
    response = client.converse(modelId=cfg["model"], messages=[{"role": "user", "content": [{"text": PROMPT}]}])
    blocks = response["output"]["message"]["content"]  # a model may also return non-text blocks
    return "".join(block["text"] for block in blocks if "text" in block)


def gcp_client(cfg):
    from google import genai

    # `enterprise=True` selects the Gemini Enterprise Agent Platform; credentials: Application Default Credentials.
    return genai.Client(enterprise=True, project=cfg["project"], location=cfg["location"])


def ask_gcp(client, cfg):
    return client.models.generate_content(model=cfg["model"], contents=PROMPT).text or ""


CLIENT_FACTORIES = {"azure": azure_client, "aws": aws_client, "gcp": gcp_client}  # tests pass fakes instead
ASK = {"azure": ask_azure, "aws": ask_aws, "gcp": ask_gcp}


# ---------- the 13-layer map: official names and status, checked 2026-10-24 ----------

Entry = namedtuple("Entry", "name status url")

_MS = "https://learn.microsoft.com/en-us/"
_AWS = "https://docs.aws.amazon.com/"
_GC = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
_AC = _AWS + "bedrock-agentcore/latest/devguide/"

STACK = [  # (layer, Azure, AWS, Google Cloud)
    ("1 Model catalog",
     Entry("Foundry Models", "GA (managed-compute deployment type: Preview)",
           _MS + "azure/foundry/concepts/foundry-models-overview"),
     Entry("Amazon Bedrock", "No status label",
           _AWS + "bedrock/latest/userguide/what-is-bedrock.html"),
     Entry("Model Garden on Gemini Enterprise Agent Platform (formerly Vertex AI Model Garden)",
           "No status label (some partner models: Preview)", _GC + "models")),
    ("2 Agent runtime",
     Entry("Foundry Agent Service: prompt agents and hosted agents",
           "GA (hosted agents GA announced 2026-07-09; voice-based prompt agents and workflow agents: Preview)",
           _MS + "azure/foundry/agents/overview"),
     Entry("Amazon Bedrock AgentCore Runtime, plus the AgentCore harness",
           "Runtime GA 2025-10-13; harness GA mid-2026 (sources give 2026-06-17 and July 2026)",
           _AC + "what-is-bedrock-agentcore.html"),
     Entry("Agent Runtime on Gemini Enterprise Agent Platform (formerly Agent Engine)",
           "GA since 2025-03-04 as Agent Engine; the current page shows no label; Managed Agents API: Preview",
           _GC + "build/runtime")),
    ("3 First-party agent SDK",
     Entry("Microsoft Agent Framework", "1.0 GA 2026-04-03 (Python and .NET); Go: Preview",
           _MS + "agent-framework/overview/"),
     Entry("Strands Agents SDK", "No status label (PyPI classifier: Production/Stable)",
           "https://github.com/strands-agents/harness-sdk"),
     Entry("Agent Development Kit (ADK)", "TypeScript 2.0: GA; Python, Go and Java: no status label",
           _GC + "build/adk")),
    ("4 Managed RAG",
     Entry("Foundry IQ: knowledge bases and agentic retrieval", "Partial GA (API: GA; portal: Preview)",
           _MS + "azure/foundry/agents/concepts/what-is-foundry-iq"),
     Entry("Amazon Bedrock Knowledge Bases: Managed Knowledge Base",
           "Managed Knowledge Base GA (July 2026 release notes); the Knowledge Bases page shows no label",
           _AWS + "bedrock/latest/userguide/knowledge-base.html"),
     Entry("RAG Engine on Gemini Enterprise Agent Platform",
           "GA since 2024-12-20; the overview page shows no label; new projects need an allowlist",
           _GC + "build/rag-engine/rag-overview")),
    ("5 Vector store",
     Entry("Azure AI Search: vector search", "No status label",
           _MS + "azure/search/vector-search-overview"),
     Entry("Amazon S3 Vectors", "GA since 2025-12-02",
           _AWS + "AmazonS3/latest/userguide/s3-vectors.html"),
     Entry("Vector Search, and Agent Retrieval (formerly Vector Search 2.0)",
           "Vector Search 2.0 GA 2026-03-05; the Agent Retrieval page shows no label",
           _GC + "build/vector-search/overview")),
    ("6 Embedding models",
     Entry("Embedding models sold by Azure: OpenAI text-embedding-3-large and -small, Cohere embed-v-4-0",
           "Listed, no status label (no Microsoft-authored text embedding model on the page checked)",
           _MS + "azure/foundry/foundry-models/concepts/models-sold-directly-by-azure"),
     Entry("Amazon Titan Text Embeddings V2; Amazon Nova Multimodal Embeddings",
           "Nova Multimodal Embeddings GA 2025-10-28; Titan: no status label",
           _AWS + "bedrock/latest/userguide/titan-embedding-models.html"),
     Entry("gemini-embedding-001 and gemini-embedding-2",
           "gemini-embedding-2 GA (2026-04-22 release notes); gemini-embedding-001: no status label",
           _GC + "models/embeddings/get-text-embeddings")),
    ("7 Agent memory",
     Entry("Memory in Foundry Agent Service", "Preview",
           _MS + "azure/foundry/agents/concepts/what-is-memory"),
     Entry("Amazon Bedrock AgentCore Memory", "GA since 2025-10-13",
           _AC + "memory.html"),
     Entry("Memory Bank and Sessions on Gemini Enterprise Agent Platform (formerly Agent Engine Memory Bank)",
           "GA per release notes (exact day not pinned, around January 2026); the product page shows no label",
           _GC + "scale/memory-bank")),
    ("8 Tools and MCP",
     Entry("Toolbox in Microsoft Foundry", "GA (tool search and Skills: Preview)",
           _MS + "azure/foundry/agents/concepts/toolbox-overview"),
     Entry("Amazon Bedrock AgentCore Gateway", "GA since 2025-10-13",
           _AC + "gateway.html"),
     Entry("Google-managed remote MCP servers; Agent Registry; Agent Gateway",
           "Agent Registry and Agent Gateway: GA 2026-06-18; managed MCP servers: status unclear (GA or Preview, no split given)",
           "https://cloud.google.com/blog/products/ai-machine-learning/google-managed-mcp-servers-are-available-for-everyone")),
    ("9 Guardrails and content safety",
     Entry("Guardrails and controls in Microsoft Foundry", "GA for models; Preview for agents",
           _MS + "azure/foundry/guardrails/guardrails-overview"),
     Entry("Amazon Bedrock Guardrails; AgentCore Policy", "Guardrails: no status label; AgentCore Policy: GA 2026-03-03",
           _AWS + "bedrock/latest/userguide/guardrails.html"),
     Entry("Model Armor", "Overview page: no status label; Model Armor for Agent Gateway: GA 2026-06-24",
           "https://docs.cloud.google.com/model-armor/overview")),
    ("10 Observability and tracing",
     Entry("Foundry Observability", "Tracing: GA for prompt and hosted agents, Preview for workflow and external agents",
           _MS + "azure/foundry/concepts/observability"),
     Entry("Amazon Bedrock AgentCore Observability", "GA since 2025-10-13",
           _AC + "observability.html"),
     Entry("Agent Observability on Gemini Enterprise Agent Platform", "GA 2026-06-18 (Preview from 2026-04-22)",
           _GC + "release-notes")),
    ("11 Evaluation",
     Entry("Evaluation in Microsoft Foundry", "GA (some evaluators: Preview)",
           _MS + "azure/foundry/concepts/evaluation-overview"),
     Entry("Amazon Bedrock AgentCore Evaluations; Amazon Bedrock evaluations",
           "AgentCore Evaluations: GA 2026-03-31; Bedrock evaluations: no status label",
           _AC + "evaluations.html"),
     Entry("Gen AI evaluation service (Gen AI evals)",
           "Status unclear (overview page shows no label; agent evaluation status not verified)",
           _GC + "models/evaluation-overview")),
    ("12 Workflow and automation",
     Entry("Microsoft Agent Framework workflows; Foundry Routines; Foundry workflows",
           "Agent Framework 1.0 GA 2026-04-03; Routines: status unclear (a GA table says GA, a July 2026 post says public preview); Foundry workflows: Preview, retiring 2026-12-01",
           _MS + "azure/foundry/agents/concepts/workflow"),
     Entry("AWS Step Functions AgentCore InvokeHarness integration; Amazon Bedrock Flows", "No status label",
           _AWS + "step-functions/latest/dg/connect-bedrockagentcore.html"),
     Entry("ADK workflow agents and graph workflows (ADK 2.0)", "TypeScript 2.0: GA; Python: no status label",
           _GC + "build/adk")),
    ("13 Agent identity and governance",
     Entry("Microsoft Entra Agent ID; Foundry control plane",
           "Entra Agent ID: GA (April 2026 release notes); Foundry control plane: Preview",
           _MS + "entra/agent-id/what-is-microsoft-entra-agent-id"),
     Entry("Amazon Bedrock AgentCore Identity; AgentCore Policy; AWS Agent Registry",
           "Identity: GA 2025-10-13; Policy: GA 2026-03-03; Registry: GA August 2026",
           _AC + "identity.html"),
     Entry("Agent Identity; Agent Registry; Agent Gateway",
           "Agent Identity: status unclear (GA on 2026-04-22 per release notes, but its API is Preview as of 2026-06-18); Registry and Gateway: GA 2026-06-18",
           _GC + "govern")),
]


def print_map():
    """Print the 13 layers. No network call: it only reads STACK."""
    for layer, *entries in STACK:
        print(f"\n{layer}")
        for cloud, entry in zip(("Azure", "AWS", "Google Cloud"), entries):
            text = f"{cloud:<14}{entry.name} | {entry.status}"
            print(textwrap.fill(text, width=110, initial_indent="  ", subsequent_indent=" " * 16))


# ---------- run ----------

def load_settings(cloud, env):
    """Return (settings, missing variable names) for one cloud. A blank value counts as missing."""
    names = SETTINGS[cloud]
    missing = [var for var in names.values() if not env.get(var, "").strip()]
    return {key: env[var].strip() for key, var in names.items()} if not missing else {}, missing


def main(argv=None, env=os.environ, client_factories=CLIENT_FACTORIES):
    parser = argparse.ArgumentParser(description="Send one prompt to a chat model on Azure, AWS and Google Cloud.")
    parser.add_argument("--cloud", choices=[*SETTINGS, "all"], default="all", help="which cloud to call (default: all)")
    parser.add_argument("--map", action="store_true", help="print the 13-layer stack table and exit; no network call")
    args = parser.parse_args(argv)  # an invalid argument exits with code 2
    if args.map:
        print_map()
        return 0

    clouds = list(SETTINGS) if args.cloud == "all" else [args.cloud]
    ready = {}
    for cloud in clouds:
        cfg, missing = load_settings(cloud, env)
        if not missing:
            ready[cloud] = cfg
        elif args.cloud == "all":
            print(f"Skipping {cloud}: set {', '.join(missing)} (see .env.example).")
        else:
            print(f"Missing configuration for {cloud}: set {', '.join(missing)} (see .env.example).")
            return 2
    if not ready:
        print("No cloud is configured. Set the variables for at least one cloud (see .env.example).")
        return 2

    print(f"Prompt: {PROMPT}")
    failed = 0
    for cloud, cfg in ready.items():
        print(f"\n== {TITLES[cloud]} ==")
        try:
            reply = ASK[cloud](client_factories[cloud](cfg), cfg).strip()
            print(reply or "(empty reply)")
        except Exception as exc:  # one cloud failing must not stop the others
            failed += 1
            print(f"ERROR {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
