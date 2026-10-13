# 8 types of RAG, and when to use each

A small Python script that runs four of the eight RAG types from the LinkedIn post "8 types of RAG, and when to use each"
on the same tiny knowledge base, so you can see where each one wins and where it misses.

LinkedIn post: (link added after publishing)

> **Not run against a live Azure resource by the author.** The SDK calls follow the official
> [Azure OpenAI v1 API guide](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)
> (`OpenAI` client, `base_url` ending in `/openai/v1/`, Entra ID token provider) and are unit-tested against fakes.

## Bring your own Azure resource

You need:

- an Azure OpenAI or Microsoft Foundry resource with two model deployments: a chat model and an embedding model
  (for example `text-embedding-3-small`)
- the `Cognitive Services OpenAI User` role on that resource for your account
- the Azure CLI, signed in with `az login` (Microsoft Entra ID sign-in, **no API keys**)
- [uv](https://docs.astral.sh/uv/) and Python 3.10+

Nothing in this repo holds credentials. Your endpoint and deployment names stay in your own `.env` (git-ignored).

## Setup and run

```bash
cp .env.example .env        # then edit .env: your endpoint and two deployment names
az login
uv run --env-file .env python rag_types.py
```

Without the three variables the script prints what to set and exits with code 2.
A full run makes about 30 small model calls (8 documents, 3 questions, 4 RAG types).

## Post to code

| Post | What the script does | Code |
|---|---|---|
| 0 CAG (skip retrieval) | Puts all 8 documents in the prompt. No index at all | `cag()` |
| 1 Naive RAG | Embeds the question, takes the 2 nearest documents by cosine similarity | `naive_rag()`, `vector_rank()` |
| 2 Hybrid + rerank | Runs keyword search (BM25) and vector search, fuses both lists with Reciprocal Rank Fusion | `hybrid_rag()`, `bm25_rank()`, `rrf()` |
| 4 Corrective RAG | Hybrid search, then the model grades the chunks. On NO, it rewrites the question and searches again | `corrective_rag()` |

The three questions are picked to show the differences:

| Question | What to look for |
|---|---|
| What does error E-4471 mean? | An exact code. Keyword search finds it every time; vector search may not |
| How long can I work from abroad? | The document says "another country", not "abroad". Vector search bridges the wording |
| My laptop is toast, what now? | Vague slang. Corrective RAG can rewrite it into a better search |

Not in the script, to keep it short: reranking with a cross-encoder or the Azure AI Search semantic ranker (type 2),
query rewriting on its own (type 3), multimodal (type 5), GraphRAG (type 6) and agentic retrieval (type 7).
In production on Azure, Azure AI Search runs hybrid search with RRF and semantic reranking in one request, and
Foundry IQ knowledge bases add agentic retrieval. See [RAG in Azure AI Search](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview).

## Tests

```bash
uv run pytest -q
```

They use fake embeddings and a fake model: no network, no Azure, no credentials. They cover BM25 finding an exact
code, RRF fusion (including empty lists), hybrid search keeping a document that vector search ranked last, corrective
RAG rewriting on NO and not on YES, CAG putting every document in the prompt, missing configuration (exit 2) and a
full run of all four types.

```text
............                                                             [100%]
12 passed in 0.01s
```

All data is fictional.
