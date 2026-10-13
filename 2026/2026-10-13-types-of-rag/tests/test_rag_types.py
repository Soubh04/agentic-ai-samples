"""Offline tests: fake embeddings and a fake model. No network, no Azure, no credentials."""

from types import SimpleNamespace

import rag_types as rt

CFG = {"endpoint": "https://example.invalid", "chat": "chat-model", "embed": "embed-model"}
ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.invalid",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat-model",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embed-model",
}


class FakeClient:
    """Stands in for the OpenAI client. Embeddings come from `vectors`; answers come from `replies` in order."""

    def __init__(self, vectors=None, replies=None):
        self.vectors = vectors or {}
        self.replies = list(replies or [])
        self.prompts = []
        self.embedded = []
        self.embeddings = SimpleNamespace(create=self._embed)
        self.responses = SimpleNamespace(create=self._respond)

    def _embed(self, model, input):
        self.embedded.extend(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vectors.get(t, [0.0, 0.0, 1.0])) for t in input])

    def _respond(self, model, input):
        self.prompts.append(input)
        text = self.replies.pop(0) if self.replies else "answer [kb-1]"
        return SimpleNamespace(output_text=text)


def doc_vecs(best=None):
    """Every doc points one way; `best` points the same way as the fake query vector [1, 0, 0]."""
    vecs = {d: [0.0, 1.0, 0.0] for d in rt.DOCS}
    if best:
        vecs[best] = [1.0, 0.0, 0.0]
    return vecs


def test_bm25_finds_exact_error_code():
    assert rt.bm25_rank("What does error E-4471 mean?", rt.DOCS)[0] == "kb-1"


def test_bm25_returns_nothing_without_shared_terms():
    assert rt.bm25_rank("zzz qqq", rt.DOCS) == []


def test_rrf_rewards_docs_found_by_both_searches():
    fused = rt.rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"  # rank 1 and rank 2 beats rank 3 and rank 1
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_lists():
    assert rt.rrf([[], []]) == []


def test_vector_rank_puts_most_similar_first():
    assert rt.vector_rank([1.0, 0.0, 0.0], doc_vecs(best="kb-3"))[0] == "kb-3"


def test_hybrid_keeps_the_exact_code_that_vector_search_missed():
    question = "What does error E-4471 mean?"
    client = FakeClient(vectors={question: [1.0, 0.0, 0.0]})
    vecs = doc_vecs(best="kb-8")  # vector search prefers the wrong certificate doc
    vecs["kb-1"] = [-1.0, 0.0, 0.0]  # and ranks the right one last
    naive_hits, _ = rt.naive_rag(client, CFG, question, vecs)
    hybrid_hits, _ = rt.hybrid_rag(client, CFG, question, vecs)
    assert "kb-1" not in naive_hits
    assert "kb-1" in hybrid_hits


def test_corrective_rag_rewrites_and_searches_again_when_graded_no():
    question = "My laptop is toast, what now?"
    client = FakeClient(replies=["NO", "laptop damaged hardware ticket", "Raise a hardware ticket [kb-4]"])
    hits, text = rt.corrective_rag(client, CFG, question, doc_vecs())
    assert client.embedded == [question, "laptop damaged hardware ticket"]  # second search used the rewrite
    assert hits[0] == "kb-4"
    assert text.endswith("[kb-4]")


def test_corrective_rag_keeps_first_results_when_graded_yes():
    question = "How long can I work from abroad?"
    client = FakeClient(vectors={question: [1.0, 0.0, 0.0]}, replies=["YES", "20 working days [kb-3]"])
    hits, _ = rt.corrective_rag(client, CFG, question, doc_vecs(best="kb-3"))
    assert hits[0] == "kb-3"
    assert client.embedded == [question]  # no second search


def test_cag_puts_every_document_in_the_prompt():
    client = FakeClient()
    hits, _ = rt.cag(client, CFG, "anything")
    assert hits == list(rt.DOCS)
    assert all(text in client.prompts[0] for text in rt.DOCS.values())


def test_missing_configuration_exits_2_with_a_friendly_message(capsys):
    assert rt.main(env={}) == 2
    assert "Bring your own" in capsys.readouterr().out


def test_load_config_strips_trailing_slash():
    cfg, missing = rt.load_config({**ENV, "AZURE_OPENAI_ENDPOINT": "https://example.invalid/"})
    assert missing == [] and cfg["endpoint"] == "https://example.invalid"


def test_main_runs_all_four_types_for_every_question(capsys):
    assert rt.main(env=ENV, client=FakeClient()) == 0
    out = capsys.readouterr().out
    assert out.count("Q: ") == len(rt.QUESTIONS)
    for name in ("0 CAG", "1 Naive", "2 Hybrid", "4 Corrective"):
        assert out.count(name) == len(rt.QUESTIONS)
