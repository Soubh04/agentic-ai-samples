# Azure vs AWS vs Google Cloud: the agent stack

A small Python script that goes with the LinkedIn post "Azure vs AWS vs Google Cloud: the agent stack, layer by layer".
The post compares the managed AI agent stack on the three clouds. The script does two things. It sends the same prompt to
one chat model on each cloud and prints the three replies, which is the model layer, the only layer you can call in a few
lines. And `--map` prints the full 13-layer stack table with each cloud's official names and status.

LinkedIn post: (link added after publishing)

> **Not run against live Azure, AWS or Google Cloud resources by the author.** The SDK calls follow the vendors' own
> examples ([Azure OpenAI v1 API with the Responses API](https://learn.microsoft.com/en-us/azure/foundry/openai/supported-languages),
> [Amazon Bedrock Converse](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html),
> [Google Gen AI SDK on the Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/sdks/overview))
> and are unit-tested against fakes. The AWS request and reply are also checked against botocore's own Converse service
> model. The three real client factories are tested for wiring only. Nothing here has talked to a real model.

## Bring your own Azure, AWS and Google Cloud

You need at least one of the three clouds. A cloud whose settings are missing is skipped when you run all three. Nothing in
this repo holds credentials: your settings stay in your own `.env` (git-ignored), and sign-in is done with each cloud's
command line tool. **No API keys.**

### Azure

- A Microsoft Foundry or Azure OpenAI resource with a chat model deployed. Note the deployment name.
- The `Cognitive Services OpenAI User` role on that resource for your account.
- The Azure CLI, signed in with `az login` (Microsoft Entra ID).

### AWS

- Amazon Bedrock in a Region where your model is available.
- The IAM permission `bedrock:InvokeModel`, which the Converse API requires.
- Credentials that boto3's standard chain can find. IAM Identity Center (SSO) is the usual choice: run
  `aws configure sso` once, then `aws sso login --profile <your-profile>` and `export AWS_PROFILE=<your-profile>`.
  Plain `aws configure` also works. The script never reads or stores keys itself.
- Model access on Bedrock. Per the AWS guide [Request access to models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html),
  access to many models is on by default: pick the model in the Bedrock model catalog and open it in the playground,
  or just invoke it. For a third-party model, the first invocation starts a subscription in the background. That needs
  the `aws-marketplace:Subscribe`, `aws-marketplace:Unsubscribe` and `aws-marketplace:ViewSubscriptions` permissions and a
  valid payment method on the account. Anthropic models also need the one-time first-time-use form. If a prerequisite is
  missing you get `AccessDeniedException`, and after you fix it the subscription can take a couple of minutes.
- A model ID or inference profile ID for `BEDROCK_MODEL_ID`. AWS's own pages show different Claude IDs in different places,
  so copy yours from the model's card in the Bedrock console.

### Google Cloud

- A Google Cloud project where you can use Gemini models on the Gemini Enterprise Agent Platform (formerly Vertex AI).
- The `Agent Platform User` role (`roles/aiplatform.user`) on that project for your account. See the
  [IAM roles reference](https://docs.cloud.google.com/iam/docs/roles-permissions/aiplatform).
- The `gcloud` CLI, with Application Default Credentials created by `gcloud auth application-default login`.
- A location for `GOOGLE_CLOUD_LOCATION` (the Google quickstart uses `global`) and a Gemini model ID for `GEMINI_MODEL`.
  Model IDs change quickly, so pick one from the Model Garden.

You also need [uv](https://docs.astral.sh/uv/) and Python 3.10 or later.

## Setup and run

```bash
cp .env.example .env        # then edit .env: fill in the clouds you want; unedited <placeholders> are skipped
az login                    # Azure
aws sso login --profile <your-profile> && export AWS_PROFILE=<your-profile>   # AWS
gcloud auth application-default login                                         # Google Cloud
uv run --env-file .env python one_prompt_three_clouds.py
```

| Command | What it does |
|---|---|
| `... one_prompt_three_clouds.py` | Same as `--cloud all`: calls every cloud whose settings are present and prints why it skipped the others |
| `... --cloud azure` (or `aws`, `gcp`) | Calls one cloud. Missing settings for that cloud exit with code 2 |
| `... --map` | Prints the 13-layer table. No network call, no settings needed |

Exit codes: 0 ok, 1 at least one cloud call failed (the other clouds still run and print), 2 missing configuration,
an invalid argument, or no cloud configured at all. A run makes one short model call per configured cloud, billed by
that cloud. Each reply is printed under a heading naming the cloud and the API used.

### Which `google-genai` flag, and why

The Google client is created with `genai.Client(enterprise=True, project=..., location=...)`. The current Google
documentation and the `google-genai` README use `enterprise=True` for the Gemini Enterprise Agent Platform. The released
package accepts it: in `google-genai` 2.33.0 (the latest on PyPI on 2026-10-24) the `Client` signature has both `enterprise` and
`vertexai`, and the docstring calls `vertexai` the "legacy flag for `enterprise`". Passing both with conflicting values
raises a `ValueError`. So the older `vertexai=True` still works but is the previous spelling. `pyproject.toml` requires
`google-genai>=2.33,<3`, and a test fails if a future release drops the `enterprise` parameter.

## Post to code

| Post | What the script does | Code |
|---|---|---|
| 1 Models: Foundry Models, Amazon Bedrock, Model Garden | Sends one prompt to a chat model on each cloud and prints each reply | `ask_azure()`, `ask_aws()`, `ask_gcp()`, `main()` |
| 1 Models: Bedrock's Converse API gives one request shape across models | The AWS call is a single `converse` request with `modelId` and `messages`. Changing `BEDROCK_MODEL_ID` changes the model, not the request | `ask_aws()` |
| 2 to 9: runtime, SDK, knowledge, memory, tools and MCP, guardrails, traces and evals, identity | Not called. `--map` prints each layer with the official name, status and a source link for the three clouds | `STACK`, `print_map()` |
| Official names as of October 2026, and the renames | Every name in `STACK` is the current one, with its status and official page, and a test keeps the table below in step with `STACK` | `STACK` |
| The mistake: picking a cloud by its model list | The model call is the easy part to move: three request shapes and three reply paths, each in one short function. The runtime contract, the identity model and where your data sits are what the post says is hard to move, and the script does not run them | `ask_azure()`, `ask_aws()`, `ask_gcp()` |

The post groups the 13 layers into nine sections: "Knowledge" covers layers 4 and 5, and "Traces and evals" covers layers 10
and 11. Layer 6 (embedding models) and layer 12 (workflow and automation) are in the table only.

## The 13-layer stack, layer by layer

Names, status and links as checked on 2026-10-24. Each link is an official vendor page: documentation, release notes or
the vendor's own blog. Some status wording and dates come from the vendor's release notes, What's New posts or a
Microsoft GA table rather than from the linked page, so check the vendor pages before you rely on a status.

- **GA** and **Preview** are the vendor's own words.
- **No status label** means the vendor page shows neither word. It does not mean the product is in preview.
- **Status unclear** means the vendor's pages contradict each other or I could not confirm it.

| Layer | Azure | AWS | Google Cloud |
|---|---|---|---|
| 1 Model catalog | [Foundry Models](https://learn.microsoft.com/en-us/azure/foundry/concepts/foundry-models-overview)<br>GA (managed-compute deployment type: Preview) | [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)<br>No status label | [Model Garden on Gemini Enterprise Agent Platform (formerly Vertex AI Model Garden)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models)<br>No status label (some partner models: Preview) |
| 2 Agent runtime | [Foundry Agent Service: prompt agents and hosted agents](https://learn.microsoft.com/en-us/azure/foundry/agents/overview)<br>GA (hosted agents GA announced 2026-07-09; voice-based prompt agents and workflow agents: Preview) | [Amazon Bedrock AgentCore Runtime, plus the AgentCore harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)<br>Runtime GA 2025-10-13; harness GA mid-2026 (sources give 2026-06-17 and July 2026) | [Agent Runtime on Gemini Enterprise Agent Platform (formerly Agent Engine)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime)<br>GA since 2025-03-04 as Agent Engine; the current page shows no label; Managed Agents API: Preview |
| 3 First-party agent SDK | [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/)<br>1.0 GA 2026-04-03 (Python and .NET); Go: Preview | [Strands Agents SDK](https://github.com/strands-agents/harness-sdk)<br>No status label (PyPI classifier: Production/Stable) | [Agent Development Kit (ADK)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)<br>TypeScript 2.0: GA; Python, Go and Java: no status label |
| 4 Managed RAG | [Foundry IQ: knowledge bases and agentic retrieval](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/what-is-foundry-iq)<br>Partial GA (API: GA; portal: Preview) | [Amazon Bedrock Knowledge Bases: Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)<br>Managed Knowledge Base GA (July 2026 release notes); the Knowledge Bases page shows no label | [RAG Engine on Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/rag-overview)<br>GA since 2024-12-20; the overview page shows no label; new projects need an allowlist |
| 5 Vector store | [Azure AI Search: vector search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)<br>No status label | [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)<br>GA since 2025-12-02 | [Vector Search, and Agent Retrieval (formerly Vector Search 2.0)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/vector-search/overview)<br>Vector Search 2.0 GA 2026-03-05; the Agent Retrieval page shows no label |
| 6 Embedding models | [Embedding models sold by Azure: OpenAI text-embedding-3-large and -small, Cohere embed-v-4-0](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure)<br>Listed, no status label (no Microsoft-authored text embedding model on the page checked) | [Amazon Titan Text Embeddings V2; Amazon Nova Multimodal Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)<br>Nova Multimodal Embeddings GA 2025-10-28; Titan: no status label | [gemini-embedding-001 and gemini-embedding-2](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings)<br>gemini-embedding-2 GA (2026-04-22 release notes); gemini-embedding-001: no status label |
| 7 Agent memory | [Memory in Foundry Agent Service](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/what-is-memory)<br>Preview | [Amazon Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html)<br>GA since 2025-10-13 | [Memory Bank and Sessions on Gemini Enterprise Agent Platform (formerly Agent Engine Memory Bank)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank)<br>GA per release notes (exact day not pinned, around January 2026); the product page shows no label |
| 8 Tools and MCP | [Toolbox in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/toolbox-overview)<br>GA (tool search and Skills: Preview) | [Amazon Bedrock AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html)<br>GA since 2025-10-13 | [Google-managed remote MCP servers; Agent Registry; Agent Gateway](https://cloud.google.com/blog/products/ai-machine-learning/google-managed-mcp-servers-are-available-for-everyone)<br>Agent Registry and Agent Gateway: GA 2026-06-18; managed MCP servers: status unclear (GA or Preview, no split given) |
| 9 Guardrails and content safety | [Guardrails and controls in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/guardrails/guardrails-overview)<br>GA for models; Preview for agents | [Amazon Bedrock Guardrails; AgentCore Policy](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)<br>Guardrails: no status label; AgentCore Policy: GA 2026-03-03 | [Model Armor](https://docs.cloud.google.com/model-armor/overview)<br>Overview page: no status label; Model Armor for Agent Gateway: GA 2026-06-24 |
| 10 Observability and tracing | [Foundry Observability](https://learn.microsoft.com/en-us/azure/foundry/concepts/observability)<br>Tracing: GA for prompt and hosted agents, Preview for workflow and external agents | [Amazon Bedrock AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)<br>GA since 2025-10-13 | [Agent Observability on Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/release-notes)<br>GA 2026-06-18 (Preview from 2026-04-22) |
| 11 Evaluation | [Evaluation in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-overview)<br>GA (some evaluators: Preview) | [Amazon Bedrock AgentCore Evaluations; Amazon Bedrock evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html)<br>AgentCore Evaluations: GA 2026-03-31; Bedrock evaluations: no status label | [Gen AI evaluation service (Gen AI evals)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview)<br>Status unclear (overview page shows no label; agent evaluation status not verified) |
| 12 Workflow and automation | [Microsoft Agent Framework workflows; Foundry Routines; Foundry workflows](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/workflow)<br>Agent Framework 1.0 GA 2026-04-03 (workflows not labelled separately); Routines: status unclear (a GA table says GA, a July 2026 post says public preview); Foundry workflows: Preview, retiring 2026-12-01 | [AWS Step Functions AgentCore InvokeHarness integration; Amazon Bedrock Flows](https://docs.aws.amazon.com/step-functions/latest/dg/connect-bedrockagentcore.html)<br>No status label | [ADK workflow agents and graph workflows (ADK 2.0)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)<br>TypeScript 2.0: GA; Python: no status label |
| 13 Agent identity and governance | [Microsoft Entra Agent ID; Foundry control plane](https://learn.microsoft.com/en-us/entra/agent-id/what-is-microsoft-entra-agent-id)<br>Entra Agent ID: GA (April 2026 release notes); Foundry control plane: Preview | [Amazon Bedrock AgentCore Identity; AgentCore Policy; AWS Agent Registry](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity.html)<br>Identity: GA 2025-10-13; Policy: GA 2026-03-03; Registry: GA August 2026 | [Agent Identity; Agent Registry; Agent Gateway](https://docs.cloud.google.com/gemini-enterprise-agent-platform/govern)<br>Agent Identity: status unclear (GA on 2026-04-22 per release notes, but its API is Preview as of 2026-06-18); Registry and Gateway: GA 2026-06-18 |

### Renamed

Older articles and SDK pages still use the old names.

- **Vertex AI** is now **Gemini Enterprise Agent Platform**, short form "Agent Platform". Google's release notes date the
  rename 2026-04-22. See [Vertex AI name changes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes).
  **Vertex AI Agent Engine** is now **Agent Runtime** (Memory Bank and Sessions keep their names), and the API resource is
  still named `ReasoningEngine`.
- **Azure AI Studio and Azure AI Foundry** are now **Microsoft Foundry**. Microsoft's
  [What is Foundry](https://learn.microsoft.com/en-us/azure/foundry/what-is-foundry) page lists the old names as
  "previous" and gives no date.

## What the script leaves out

- Layers 2 to 13. The script calls only the model layer. It deploys no agent, and uses no memory, knowledge base, tool
  gateway, guardrail, trace, evaluation or identity product.
- A comparison of the models. Replies differ because the models you configure differ. This is not a benchmark, and it
  measures no latency or price.
- That Claude runs on all three clouds. The post says so, and the script sends whatever model IDs you configure.
- Other API choices. Microsoft's docs recommend the Responses API for new work, so the Azure call uses it. Bedrock's
  `bedrock-runtime` also offers Messages, Responses, Chat Completions and Invoke, and AWS's docs do not name one as
  recommended, so the AWS call uses Converse as the post does. Microsoft's `azure-ai-projects` client is a Foundry-native
  alternative to the plain `openai` client.
- Request options such as a system prompt, token limits, temperature, streaming and tool calls.
- Non-text reply parts. The AWS call joins the text blocks and ignores the rest (for example reasoning blocks).
- Anything the table does not say: prices, regions, quotas, and the contract details of each runtime.
- The table is a snapshot of one day. Product names, status and model IDs move fast, so check the vendor pages.

## Tests

```bash
uv run pytest -q
```

They run offline with fake clients, so no network and no credentials. Every test runs with network connections blocked.
They cover the request each adapter sends, reply extraction for each cloud including an empty reply and non-text blocks, a
regression test per adapter that fails if the reply field path is wrong, the three real client factories, an AWS call
checked against botocore's Converse service model, a named cloud with missing settings exiting with code 2, `all` skipping
unconfigured clouds and running the rest, one cloud's error not stopping the others, `--map` printing 13 layers with three
non-empty entries and an https source each, and the README table matching the data.

```text
..................................                                       [100%]
34 passed in 0.66s
```

All data is fictional.
