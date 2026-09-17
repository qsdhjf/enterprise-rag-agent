"""Offline regression coverage for uploaded-file Agent, index jobs and setup."""

import json
from importlib.resources import files
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.cli.evaluate_knowledge import Fixture, run_fixture
from enterprise_agent.infrastructure.llm_client import LLMResult
from enterprise_agent.infrastructure.reranker_client import RerankerScore
from enterprise_agent.knowledge.agent import run_library_agent
from enterprise_agent.knowledge.profiles import Provider, ProfileConfig, ProfileStore
from enterprise_agent.knowledge.retrieval import LibraryRetriever, corpus_hash, normalized, fuse, row_key
from enterprise_agent.knowledge.service import KnowledgeService
from enterprise_agent.knowledge.store import KnowledgeError, KnowledgeStore
from enterprise_agent.web.api import app


def provider(**kwargs):
    return Provider(base_url="http://127.0.0.1:19999/v1", model="synthetic-model", dimensions=2, **kwargs)


def insert(store, owner, library, name, text):
    document = store.put_document(owner, library, name, text.encode())
    store.finish(document["id"], 1, {"chunks": [{"ordinal": 1, "location": "第 1 行", "text": text}], "warnings": []})
    return document["id"]


@pytest.fixture
def service(tmp_path):
    value = KnowledgeService(tmp_path, start_worker=False)
    yield value
    value.close()


class Embedding:
    def __init__(self, *args, **kwargs):
        pass
    def embed_text(self, text):
        return [1., 0.] if any(t in text for t in ("住宿", "旅馆", "alpha")) else [0., 1.]
    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]
    def close(self):
        pass


class Reranker:
    def __init__(self, *args, **kwargs):
        pass
    def rerank(self, query, texts):
        return tuple(RerankerScore(i, 10. if "住宿" in text else 0.) for i, text in enumerate(texts))
    def close(self):
        pass


@pytest.fixture
def fake_vectors(monkeypatch):
    monkeypatch.setattr("enterprise_agent.infrastructure.embedding_client.OpenAICompatibleEmbeddingClient", Embedding)
    monkeypatch.setattr("enterprise_agent.infrastructure.reranker_client.TEIRerankerClient", Reranker)


def indexed(service):
    library = service.store.create_library("alice", "library")["id"]
    document = insert(service.store, "alice", library, "a.txt", "住宿费用上限 480 元 alpha")
    insert(service.store, "alice", library, "b.txt", "员工休假审批 beta")
    service.update_profile("embedding", provider(), 0)
    service.store.queue_vectors("alice", library, provider().fingerprint, 2)
    service.build_vectors(service.store.claim_vectors())
    assert service.store.vector_status("alice", library, provider().fingerprint)["status"] == "ready"
    return library, document


@pytest.mark.parametrize("strategy", ["bm25", "vector", "hybrid", "reranked"])
def test_real_retriever_strategies(service, fake_vectors, strategy):
    library, document = indexed(service)
    service.update_profile("reranker", provider(), 1)
    with service.retriever("alice", library, strategy) as retriever:
        rows = retriever.search("住宿" if strategy == "bm25" else "旅馆", 1, strategy)
        assert rows[0]["document_id"] == document
        assert retriever.query_embeddings == (strategy != "bm25")
        assert retriever.rerank_calls == (strategy == "reranked")


def test_vectors_reject_other_owner_and_stale_documents(service, fake_vectors):
    library, document = indexed(service)
    with pytest.raises(KnowledgeError) as error:
        service.store.read_vectors("mallory", library, provider().fingerprint, "x")
    assert error.value.status == 404
    service.store.put_document("alice", library, "a.txt", b"changed", document)
    assert service.store.vector_status("alice", library, provider().fingerprint)["status"] == "stale"
    with service.store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM vectors WHERE document_id=?", (document,)).fetchone()[0] == 0
    with service.retriever("alice", library, "hybrid") as retriever, pytest.raises(KnowledgeError):
        retriever.search("alpha", strategy="hybrid")


def test_profile_change_invalidates_index(service, fake_vectors):
    library, _ = indexed(service)
    changed = provider().model_copy(update={"model": "different"})
    service.update_profile("embedding", changed, 1)
    assert service.store.vector_status("alice", library, changed.fingerprint)["status"] == "stale"


def test_cancel_and_restart_require_new_consent(service):
    library = service.store.create_library("alice", "library")["id"]
    insert(service.store, "alice", library, "a.txt", "alpha")
    service.store.queue_vectors("alice", library, "profile")
    job = service.store.claim_vectors()
    service.store.cancel_vectors("alice", library)
    with pytest.raises(KnowledgeError):
        service.store.vector_progress(job, 1)
    service.store.queue_vectors("alice", library, "profile")
    service.store.recover()
    assert service.store.claim_vectors() is None
    assert service.store.vector_status("alice", library, "profile")["status"] == "failed"


def test_delete_during_vector_call_never_resurrects(service, monkeypatch):
    library = service.store.create_library("alice", "library")["id"]
    document = insert(service.store, "alice", library, "a.txt", "alpha")
    service.update_profile("embedding", provider(), 0)
    class DeleteDuringCall(Embedding):
        def embed_texts(self, texts):
            service.store.mutate_document("alice", library, document, "delete")
            return super().embed_texts(texts)
    monkeypatch.setattr("enterprise_agent.infrastructure.embedding_client.OpenAICompatibleEmbeddingClient", DeleteDuringCall)
    service.store.queue_vectors("alice", library, provider().fingerprint, 2)
    service.build_vectors(service.store.claim_vectors())
    with service.store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
        assert db.execute("SELECT status FROM vector_jobs").fetchone()[0] == "failed"


@pytest.mark.parametrize("vector", [[0,0], [1], [float("nan"),1], [True,1], [float("inf"),1]])
def test_invalid_vectors_fail_closed(vector):
    with pytest.raises(KnowledgeError):
        normalized(vector, 2)


def test_rrf_merges_duplicate_chunks():
    a = dict(document_id="a", ordinal=1, revision=1)
    b = dict(document_id="b", ordinal=1, revision=1)
    assert [row_key(r) for r in fuse([a,b],[b],5)] == [row_key(b),row_key(a)]


class GraphModel:
    def __init__(self, source="uploaded_file", delete=None):
        self.planners = self.judges = self.answers = 0
        self.source, self.delete = source, delete
    def generate(self, **kwargs):
        prompt = kwargs["system_prompt"]
        if "You are the planner" in prompt:
            self.planners += 1
            assert json.loads(kwargs["user_prompt"])["allowed_sources"] == ["uploaded_file"]
            payload = dict(kind="search_enterprise_knowledge", query="alpha" if self.planners==1 else "beta",
                           source=self.source, top_k=1, reason="Find the missing fact")
        elif "evidence-sufficiency judge" in prompt:
            self.judges += 1
            payload = dict(decision="partially_supported" if self.judges==1 else "supported", reason="synthetic audit",
                facts=[dict(fact="owner", status="supported", conclusion="Ada", evidence_ids=["E1"]),
                       dict(fact="date", status="missing" if self.judges==1 else "supported",
                            conclusion=None if self.judges==1 else "Oct 12", evidence_ids=[] if self.judges==1 else ["E2"])])
        else:
            self.answers += 1
            if self.delete:
                self.delete()
            return LLMResult("Ada [E1]; Oct 12 [E2].", "stub", "answer", 10, 10)
        return LLMResult(json.dumps(payload), "stub", "stub", 10, 10)


def graph_library(service):
    library = service.store.create_library("alice", "library")["id"]
    first = insert(service.store, "alice", library, "first.txt", "alpha owner is Ada")
    insert(service.store, "alice", library, "second.txt", "beta date is Oct 12")
    service.update_profile("llm", provider(), 0)
    return library, first


def test_actual_graph_two_rounds_with_owned_uploads(service):
    library, _ = graph_library(service)
    client = GraphModel()
    result = run_library_agent(service, "alice", library, "Who and when?", "test", client=client)
    assert (client.planners, client.judges, client.answers) == (2,2,1)
    assert result["outcome"] == "supported"
    assert result["citations"] == ["E1", "E2"]
    assert result["tool_call_count"] == 2
    assert result["model_token_count"] == 100
    assert [a["label"] for a in result["activity"]].count("evidence_judge") == 2
    assert all(card["source"] == "uploaded_file" for card in result["evidence"])


def test_agent_cannot_select_enterprise_index(service):
    library, _ = graph_library(service)
    client = GraphModel(source="jira")
    with pytest.raises(KnowledgeError):
        run_library_agent(service, "alice", library, "test", "test", client=client)
    assert client.judges == client.answers == 0


def test_deleted_evidence_not_returned_after_answer(service):
    library, first = graph_library(service)
    client = GraphModel(delete=lambda: service.store.mutate_document("alice", library, first, "delete"))
    with pytest.raises(KnowledgeError):
        run_library_agent(service, "alice", library, "test", "test", client=client)


def test_nonowner_or_stale_index_sends_no_prompt(service):
    library, _ = graph_library(service)
    client = GraphModel()
    with pytest.raises(KnowledgeError):
        run_library_agent(service, "mallory", library, "test", "test", client=client)
    with pytest.raises(KnowledgeError):
        run_library_agent(service, "alice", library, "test", "test", "hybrid", client=client)
    assert client.planners == 0


def test_profiles_private_atomic_revision_and_endpoint_key_rules(tmp_path):
    profiles = ProfileStore(tmp_path)
    secret = "example-private-credential"
    first = profiles.update("llm", provider(api_key=secret), 0)
    assert secret not in json.dumps(first)
    assert secret not in json.dumps(profiles.summary())
    assert "base_url" not in json.dumps(profiles.summary())
    assert ProfileStore(tmp_path).snapshot().llm.api_key.get_secret_value() == secret
    with pytest.raises(KnowledgeError):
        profiles.update("llm", provider(), 0)
    with pytest.raises(KnowledgeError):
        profiles.update("llm", provider().model_copy(update={"base_url":"https://different.example/v1"}), 1)
    profiles.update("llm", provider(), 1)
    assert profiles.snapshot().llm.api_key.get_secret_value() == secret


@pytest.mark.parametrize("url", ["http://remote.example/v1", "https://user:pass@host/v1", "https://host/v1?key=bad", "file:///tmp/model"])
def test_unsafe_provider_urls_rejected(url):
    with pytest.raises(ValueError):
        Provider(base_url=url, model="test")


@pytest.fixture
def web(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODE", "demo")
    monkeypatch.setenv("AGENT_KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SETUP_ENABLED", "1")
    monkeypatch.delenv("AGENT_KNOWLEDGE_MODEL_ENABLED", raising=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1234)) as client:
        client.get("/api/knowledge/session")
        yield client


def test_setup_roundtrip_masks_key_and_no_auto_network(web, monkeypatch):
    import httpx
    def forbidden(*a, **k):
        raise AssertionError("Unexpected network on save")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    body = dict(provider={"base_url":"https://provider.example/v1","model":"example","api_key":"private-test-secret"},revision=0)
    response = web.put("/api/knowledge/setup/llm", json=body)
    assert response.status_code == 200
    assert "private-test-secret" not in response.text
    assert web.get("/api/knowledge/session").json()["model_enabled"] is True
    assert web.post("/api/knowledge/setup/llm/test", json={"consent":False}).status_code == 422
    assert web.put("/api/knowledge/setup/llm", json=body).status_code == 409


@pytest.mark.parametrize("body", [["do-not-echo-secret"], {"provider":{"api_key":"do-not-echo-secret"},"revision":0}, {"provider":None,"revision":"do-not-echo-secret"}])
def test_bad_setup_inputs_never_echo_secrets(web, body):
    response = web.put("/api/knowledge/setup/llm", json=body)
    assert response.status_code == 422
    assert "do-not-echo-secret" not in response.text


def test_setup_rejects_csrf_and_disabled(web, monkeypatch):
    assert web.get("/api/knowledge/setup", headers={"Origin":"https://evil.example"}).status_code == 403
    monkeypatch.setenv("AGENT_SETUP_ENABLED", "0")
    assert web.get("/api/knowledge/setup").status_code == 403


def test_evaluation_fixture_reproducible_no_production_claim():
    fixture = Fixture.model_validate_json(files("enterprise_agent.knowledge").joinpath("eval_fixture.json").read_text(encoding="utf-8"))
    report = run_fixture(fixture)
    assert report["synthetic"] is True
    assert report["cases_count"] == 10
    assert report["summary"]["bm25"]["no_gold_cases"] == 2
    assert all(row["answer"] is None for row in report["answer_review_template"])
    assert "not production" in report["limitations"]


def test_v1_migration_creates_backup_and_preserves_documents(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    library = store.create_library("alice", "library")["id"]
    document = insert(store,"alice",library,"a.txt","alpha")
    with store.db(True) as db:
        db.execute("DROP TABLE vectors")
        db.execute("DROP TABLE vector_jobs")
        db.execute("PRAGMA user_version=1")
    migrated = KnowledgeStore(store.path)
    assert migrated.corpus("alice",library)[0]["document_id"] == document
    assert len(list(tmp_path.glob("before-v3-*.sqlite3"))) == 1
    with migrated.db() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
