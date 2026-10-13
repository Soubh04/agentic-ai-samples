"""Types of RAG, side by side, on one tiny knowledge base.

Four of the eight types from the post, run on the same questions:
  0 CAG         no retrieval: every document goes into the prompt
  1 Naive RAG   vector search only
  2 Hybrid RAG  keyword (BM25) + vector, fused with Reciprocal Rank Fusion
  4 Corrective  hybrid, then the model grades the chunks; a weak result triggers a rewritten search

Bring your own Azure OpenAI or Microsoft Foundry resource with a chat model and an embedding model
deployed. Sign in with `az login` (Entra ID, no API keys) and set the variables in .env.example.

    uv run --env-file .env python rag_types.py

Exit codes: 0 ok, 2 missing configuration.
"""

import math
import os
import re
import sys
from collections import Counter

DOCS = {  # fictional IT helpdesk knowledge base
    "kb-1": "Error E-4471 on the VPN client means your device certificate expired. Renew it in the Company Portal app.",
    "kb-2": "VPN connection drops are usually caused by hotel or airport Wi-Fi. Switch to a mobile hotspot.",
    "kb-3": "You may work from another country for up to 20 working days per year. Tell your manager first.",
    "kb-4": "Laptop damaged by liquid or a fall? Raise a hardware ticket. You get a loaner laptop within one working day.",
    "kb-5": "Passwords expire every 90 days. Reset yours at the self-service password portal.",
    "kb-6": "Printer error P-02 means the paper tray is open. Close the tray and retry.",
    "kb-7": "New starters receive their laptop on day one. Personal devices need Intune enrolment for email.",
    "kb-8": "Certificates for Wi-Fi are installed automatically. Do not install certificates sent by email.",
}
QUESTIONS = [
    "What does error E-4471 mean?",          # exact code: keyword search shines
    "How long can I work from abroad?",      # different words: vector search shines
    "My laptop is toast, what now?",         # vague: corrective RAG may rewrite it
]
TOP_K = 2
RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009)
ENV = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


# ---------- retrieval: plain functions, no service needed ----------

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def bm25_rank(query, docs, k1=1.5, b=0.75):
    """Keyword search. Returns doc ids, best first, only docs that share a term with the query."""
    tokens = {doc_id: tokenize(text) for doc_id, text in docs.items()}
    avg_len = sum(len(t) for t in tokens.values()) / len(tokens)
    scores = {}
    for term in set(tokenize(query)):
        df = sum(term in t for t in tokens.values())
        if df == 0:
            continue
        idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        for doc_id, t in tokens.items():
            tf = Counter(t)[term]
            if tf:
                norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg_len))
                scores[doc_id] = scores.get(doc_id, 0) + idf * norm
    return sorted(scores, key=scores.get, reverse=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def vector_rank(query_vec, doc_vecs):
    """Vector search. Returns all doc ids, most similar first."""
    return sorted(doc_vecs, key=lambda doc_id: cosine(query_vec, doc_vecs[doc_id]), reverse=True)


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over every list a doc appears in."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


# ---------- model calls: one small wrapper each, so tests can fake them ----------

def embed(client, model, texts):
    return [item.embedding for item in client.embeddings.create(model=model, input=texts).data]


def ask(client, model, prompt):
    return client.responses.create(model=model, input=prompt).output_text.strip()


def answer(client, model, question, doc_ids):
    context = "\n".join(f"[{d}] {DOCS[d]}" for d in doc_ids)
    prompt = (
        "Answer in one sentence using only the sources below and cite the source id. "
        "If they don't answer it, say you don't know.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    return ask(client, model, prompt)


# ---------- the four RAG types ----------

def cag(client, cfg, question):
    return list(DOCS), answer(client, cfg["chat"], question, list(DOCS))


def naive_rag(client, cfg, question, doc_vecs):
    hits = vector_rank(embed(client, cfg["embed"], [question])[0], doc_vecs)[:TOP_K]
    return hits, answer(client, cfg["chat"], question, hits)


def hybrid_search(client, cfg, query, doc_vecs):
    vector_hits = vector_rank(embed(client, cfg["embed"], [query])[0], doc_vecs)
    return rrf([bm25_rank(query, DOCS), vector_hits])[:TOP_K]


def hybrid_rag(client, cfg, question, doc_vecs):
    hits = hybrid_search(client, cfg, question, doc_vecs)
    return hits, answer(client, cfg["chat"], question, hits)


def corrective_rag(client, cfg, question, doc_vecs):
    hits = hybrid_search(client, cfg, question, doc_vecs)
    chunks = "\n".join(DOCS[d] for d in hits)
    grade = ask(client, cfg["chat"], f"Do these passages answer the question? Reply YES or NO only.\n\n{chunks}\n\nQuestion: {question}")
    if not grade.upper().startswith("YES"):
        rewritten = ask(client, cfg["chat"], f"Rewrite this IT helpdesk question as a short, specific search query. Reply with the query only.\n\n{question}")
        hits = hybrid_search(client, cfg, rewritten, doc_vecs)
    return hits, answer(client, cfg["chat"], question, hits)


# ---------- setup and main ----------

def load_config(env=os.environ):
    missing = [name for name in ENV if not env.get(name)]
    if missing:
        return None, missing
    return {"endpoint": env[ENV[0]].rstrip("/"), "chat": env[ENV[1]], "embed": env[ENV[2]]}, []


def make_client(endpoint):
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import OpenAI

    token = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
    return OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=token)


def main(env=os.environ, client=None):
    cfg, missing = load_config(env)
    if missing:
        print("Bring your own Azure OpenAI or Foundry resource: set " + ", ".join(missing)
              + " (see .env.example), then run `az login`.")
        return 2
    client = client or make_client(cfg["endpoint"])
    doc_vecs = dict(zip(DOCS, embed(client, cfg["embed"], list(DOCS.values()))))
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        for name, run in [
            ("0 CAG       ", lambda q: cag(client, cfg, q)),
            ("1 Naive     ", lambda q: naive_rag(client, cfg, q, doc_vecs)),
            ("2 Hybrid    ", lambda q: hybrid_rag(client, cfg, q, doc_vecs)),
            ("4 Corrective", lambda q: corrective_rag(client, cfg, q, doc_vecs)),
        ]:
            hits, text = run(question)
            sources = "all docs" if len(hits) == len(DOCS) else ", ".join(hits)
            print(f"  {name} [{sources}] {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
