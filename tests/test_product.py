"""Product contracts: no-network demo, fail-closed auth, bounded HTTP input."""

import hashlib
import json
from threading import BoundedSemaphore

import httpx
import pytest
from fastapi.testclient import TestClient

from enterprise_agent.web.api import app
from enterprise_agent.web.security import Identity, load_identities
from enterprise_agent.demo.runner import DemoRunner
from enterprise_agent.cli.evaluate import load_cases_text
from enterprise_agent.evaluation.metrics import retrieval_metrics


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "demo")
    # The real demo lifespan must not initialize any external dependencies.
    def forbidden(*args, **kwargs):
        raise AssertionError("Demo attempted an external service request")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    with TestClient(app) as session:
        yield session


def test_demo_lifespan_is_offline_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/api/config").json()["mode"] == "demo"
    ready = client.get("/health/ready").json()
    assert ready["status"] == "ready"
    assert ready["opensearch_reachable"] is False


def test_demo_returns_matching_evidence_and_no_model_tokens(client):
    response = client.post("/ask", json={"question": "Atlas 归档"})
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "demo"
    assert result["model_token_count"] == 0
    assert result["citations"] == [card["id"] for card in result["evidence"]]
    assert all(card["excerpt"] in result["answer"] for card in result["evidence"])
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"


def test_unknown_demo_query_has_no_fabricated_evidence(client):
    result = client.post("/ask", json={"question": "xylophoniumquasar"}).json()
    assert result["outcome"] == "unsupported"
    assert result["evidence"] == result["citations"] == []


@pytest.mark.parametrize("payload", [
    {"question": " "}, {"question": "x" * 2001},
    {"question": "Atlas", "tenant_id": "victim"},
])
def test_invalid_or_identity_bearing_payload_is_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_body_limit_applies_before_json_decode(client):
    assert client.post("/ask", content=b"x" * 17000).status_code == 413
    assert client.post("/ask", content=iter([b"x" * 10000, b"x" * 10000])).status_code == 413


def test_busy_request_does_not_consume_an_extra_slot(client):
    slots = BoundedSemaphore(1)
    app.state.request_slots = slots
    slots.acquire()
    assert client.post("/ask", json={"question": "Atlas"}).status_code == 429
    slots.release()
    assert client.post("/ask", json={"question": "Atlas"}).status_code == 200


def test_unexpected_dependency_failure_is_sanitized_and_releases_slot(client):
    class BrokenRunner:
        def ask(self, **kwargs):
            raise RuntimeError("private-provider-detail")
    app.state.agent_runner = BrokenRunner()
    app.state.request_slots = BoundedSemaphore(1)
    response = client.post("/ask", json={"question": "Atlas"})
    assert response.status_code == 503
    assert "private-provider-detail" not in response.text
    assert response.json()["detail"]["request_id"] == response.headers["x-request-id"]
    app.state.agent_runner = DemoRunner()
    assert client.post("/ask", json={"question": "Atlas"}).status_code == 200


def identity_record(token="test-only-token"):
    return {"token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "principal": {"tenant_id": "tenant-a", "subject_id": "alice", "roles": ["employee"],
                          "groups": [], "authz_version": "v1"},
            "retrieval_policy": {"policy_version": "v1", "allowed_sources": ["jira"]}}


def test_live_http_identity_is_server_owned(client):
    app.state.demo_mode = False
    app.state.identities = (Identity.model_validate(identity_record()),)
    captured = []
    class RecordingRunner(DemoRunner):
        def ask(self, question, runtime_context):
            captured.append(runtime_context)
            return super().ask(question, runtime_context)
    app.state.agent_runner = RecordingRunner()
    assert client.post("/ask", json={"question": "Atlas"}).status_code == 401
    assert client.get("/health/ready").status_code == 401
    assert client.get("/health/live").status_code == 200
    headers = {"Authorization": "Bearer invalid"}
    assert client.post("/ask", json={"question": "Atlas"}, headers=headers).status_code == 401
    assert not captured
    headers["Authorization"] = "Bearer test-only-token"
    assert client.post("/ask", json={"question": "Atlas"}, headers=headers).status_code == 200
    assert captured[0].principal.tenant_id == "tenant-a"
    assert captured[0].retrieval_policy.allowed_sources == frozenset({"jira"})


def test_live_startup_requires_identity_before_external_clients(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "live")
    monkeypatch.delenv("AGENT_IDENTITY_FILE", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_IDENTITY_FILE"):
        with TestClient(app):
            pass


@pytest.mark.parametrize("variant", ["empty", "duplicate", "demo_tenant"])
def test_identity_file_rejects_unsafe_configs(tmp_path, variant):
    rows = [identity_record()]
    if variant == "empty":
        rows = []
    elif variant == "duplicate":
        rows *= 2
    else:
        rows[0]["principal"]["tenant_id"] = "local-demo-tenant"
    path = tmp_path / "identities.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_identities(str(path))


def test_metrics_do_not_confuse_hit_with_recall():
    metrics = retrieval_metrics(["irrelevant", "a", "a"], {"a", "b"}, 3)
    assert metrics["hit"] == 1
    assert metrics["recall"] == .5
    assert metrics["mrr"] == .5
    assert 0 < metrics["ndcg"] < .5
    assert retrieval_metrics([], set(), 5)["recall"] is None
    assert retrieval_metrics([], set(), 5)["no_result"] == 1


def test_eval_cases_require_unambiguous_gold_and_unique_queries():
    with pytest.raises(ValueError, match="source::"):
        load_cases_text('{"case_id":"a","query":"q","relevant_document_ids":["doc"]}')
    row = '{"case_id":"a","query":"q","source":"jira","relevant_document_ids":["doc"]}'
    assert load_cases_text(row)[0].relevant_document_ids == ["jira::doc"]
    with pytest.raises(ValueError, match="unique"):
        load_cases_text(row + "\n" + row)
