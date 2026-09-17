"""Personal knowledge contracts, real parser subprocesses and hostile-input cases."""

import io
import json
import time
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.knowledge.parser import extract, validate_filename
from enterprise_agent.knowledge.service import KnowledgeService
from enterprise_agent.knowledge.store import KnowledgeError, KnowledgeStore
from enterprise_agent.web.api import app
from enterprise_agent.web.security import Identity


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODE", "demo")
    monkeypatch.setenv("AGENT_KNOWLEDGE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.delenv("AGENT_KNOWLEDGE_MODEL_ENABLED", raising=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as session:
        assert session.get("/api/knowledge/session").status_code == 200
        yield session


def library(client, name="研究资料"):
    response = client.post("/api/knowledge/libraries", json={"name": name})
    assert response.status_code == 201, response.text
    return f'/api/knowledge/libraries/{response.json()["id"]}'


def upload(client, base, name="policy.txt", text="Orchid 归档规则：审批人为陈老师，预算为 4721 元。"):
    response = client.post(base + "/documents", params={"filename": name},
                           content=text.encode() if isinstance(text, str) else text)
    assert response.status_code == 202, response.text
    return response.json()["id"]


def wait_document(client, base, identity, expected="ready"):
    for _ in range(100):
        doc = client.get(f"{base}/documents/{identity}").json()
        if doc["status"] in {"ready", "failed"}:
            assert doc["status"] == expected, doc
            return doc
        time.sleep(.05)
    pytest.fail("Background parser did not finish")


def test_upload_query_preview_delete_lifecycle(client):
    base = library(client)
    doc = upload(client, base)
    ready = wait_document(client, base, doc)
    assert ready["chunk_count"] == 1
    assert "raw" not in ready
    result = client.post(base + "/ask", json={"question": "Orchid 预算"}).json()
    assert result["mode"] == "knowledge-local"
    assert result["outcome"] == "unverified"
    assert result["model_token_count"] == 0
    assert "4721" in result["answer"]
    assert result["evidence"][0]["document_id"] == doc
    assert "第 1" in result["evidence"][0]["location"]
    assert client.delete(base + f"/documents/{doc}").status_code == 204
    assert client.get(base + f"/documents/{doc}").status_code == 404
    assert client.post(base + "/ask", json={"question": "Orchid"}).json()["evidence"] == []


def test_replace_invalidates_old_chunks_and_does_not_duplicate(client):
    base = library(client)
    doc = upload(client, base)
    wait_document(client, base, doc)
    assert client.post(base + "/documents", params={"filename": "policy.txt"}, content=b"changed").status_code == 409
    response = client.put(base + f"/documents/{doc}", params={"filename": "policy.txt"}, content=b"Helios manager is Ada")
    assert response.status_code == 202
    ready = wait_document(client, base, doc)
    assert ready["revision"] == 2
    assert client.post(base + "/ask", json={"question": "Orchid"}).json()["evidence"] == []
    assert "Ada" in client.post(base + "/ask", json={"question": "Helios manager"}).json()["answer"]
    assert len(client.get(base + "/documents").json()) == 1


def test_failed_parse_is_visible_and_retryable(client):
    base = library(client)
    doc = upload(client, base, "scanned.pdf", b"not a PDF")
    failed = wait_document(client, base, doc, "failed")
    assert failed["error"]
    assert failed["chunk_count"] == 0
    assert client.post(base + f"/documents/{doc}/retry").status_code == 202
    assert wait_document(client, base, doc, "failed")["revision"] == 2


def test_delete_library_cascades_and_inaccessible(client):
    base = library(client)
    upload(client, base)
    assert client.delete(base).status_code == 204
    assert client.get(base + "/documents").status_code == 404
    assert client.post(base + "/ask", json={"question": "Orchid"}).status_code == 404


def test_session_cookie_is_httponly_and_separate_browsers_cannot_access(client):
    base = library(client)
    doc = upload(client, base)
    other = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12346))
    try:
        result = other.get("/api/knowledge/session")
        assert "HttpOnly" in result.headers["set-cookie"]
        assert "SameSite=strict" in result.headers["set-cookie"]
        assert other.get("/api/knowledge/libraries").json() == []
        assert other.get(base + "/documents").status_code == 404
        assert other.get(base + f"/documents/{doc}").status_code == 404
        assert other.delete(base + f"/documents/{doc}").status_code == 404
        assert other.put(base + f"/documents/{doc}", params={"filename": "x.txt"}, content=b"x").status_code == 404
        assert other.post(base + f"/documents/{doc}/retry").status_code == 404
        assert other.post(base + "/ask", json={"question": "Orchid"}).status_code == 404
        assert other.delete(base).status_code == 404
    finally:
        other.close()


@pytest.mark.parametrize("headers", [{"Host": "evil.example"}, {"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"}])
def test_local_session_blocks_rebinding_and_cross_origin(client, headers):
    assert client.get("/api/knowledge/session", headers=headers).status_code == 403
    assert client.post("/api/knowledge/libraries", json={"name": "evil"}, headers=headers).status_code == 403


def test_remote_peer_cannot_spoof_localhost(client):
    other = TestClient(app, base_url="http://127.0.0.1", client=("10.0.0.1", 12346))
    try:
        assert other.get("/api/knowledge/session", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403
    finally:
        other.close()


@pytest.mark.parametrize("name", ["../a.txt", "a/b.txt", "a\\b.txt", "bad.exe", "secret.txt:stream", "bad\x00.txt", " a.txt"])
def test_unsafe_names_rejected_before_worker(client, name):
    base = library(client)
    assert client.post(base + "/documents", params={"filename": name}, content=b"text").status_code == 422


def test_upload_body_limit_does_not_raise_ask_limit(client):
    base = library(client)
    assert client.post(base + "/documents", params={"filename": "a.txt"}, content=b"a" * (8*1024**2+1)).status_code == 413
    assert client.post(base + "/documents", params={"filename": "a.txt"}, content=iter([b"a"*1024**2]*9)).status_code == 413
    assert client.post(base + "/ask", content=b"a" * 17000).status_code == 413
    assert client.post(base + "/documents", params={"filename": "a.txt"}, content=b"").status_code == 422


@pytest.mark.parametrize("body", [{"question": " "}, {"question": "a", "owner": "someone"}, {"question": "a", "answer_mode": "auto"}])
def test_question_contract_rejects_identity_override(client, body):
    assert client.post(library(client) + "/ask", json=body).status_code == 422


def test_model_is_opt_in_and_invalid_citations_fail_closed(client, monkeypatch):
    base = library(client)
    doc = upload(client, base)
    wait_document(client, base, doc)
    assert client.post(base + "/ask", json={"question": "Orchid", "answer_mode": "model"}).status_code == 409
    knowledge = app.state.knowledge
    knowledge.model_settings = object()
    calls = []
    def fake(question, evidence):
        calls.append(evidence)
        return SimpleNamespace(text=json.dumps({"answer": "预算 4721 元 [E1]", "citations": ["E1"]}), input_tokens=20, output_tokens=10)
    monkeypatch.setattr(knowledge, "_generate", fake)
    assert client.post(base + "/ask", json={"question": "Orchid"}).status_code == 200
    assert not calls
    result = client.post(base + "/ask", json={"question": "Orchid", "answer_mode": "model"}).json()
    assert result["mode"] == "knowledge-model" and result["model_token_count"] == 30
    assert result["outcome"] == "unverified" and len(calls) == 1
    monkeypatch.setattr(knowledge, "_generate", lambda *args: SimpleNamespace(text='{"answer":"made up [E999]","citations":["E999"]}'))
    assert client.post(base + "/ask", json={"question": "Orchid", "answer_mode": "model"}).status_code == 502


def test_document_deletion_during_model_call_suppresses_stale_answer(client, monkeypatch):
    base = library(client)
    doc = upload(client, base)
    wait_document(client, base, doc)
    knowledge = app.state.knowledge
    knowledge.model_settings = object()
    def generate(*args):
        assert client.delete(base + f"/documents/{doc}").status_code == 204
        return SimpleNamespace(text='{"answer":"Old evidence [E1]","citations":["E1"]}', input_tokens=1, output_tokens=1)
    monkeypatch.setattr(knowledge, "_generate", generate)
    result = client.post(base + "/ask", json={"question": "Orchid", "answer_mode": "model"})
    assert result.status_code == 409
    assert "Old evidence" not in result.text


def test_busy_question_slots_released_on_model_failure(client, monkeypatch):
    base = library(client)
    slots = app.state.request_slots
    slots.acquire(); slots.acquire()
    assert client.post(base + "/ask", json={"question": "x"}).status_code == 429
    slots.release(); slots.release()
    assert client.post(base + "/ask", json={"question": "x"}).status_code == 200


def test_authenticated_owner_uses_tenant_and_subject_not_token(client):
    import hashlib
    def identity(token, tenant, subject):
        return Identity.model_validate({"token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "principal": {"tenant_id": tenant, "subject_id": subject, "roles": ["employee"], "groups": [], "authz_version": "v1"},
            "retrieval_policy": {"policy_version": "v1", "allowed_sources": ["github"]}})
    app.state.demo_mode = False
    app.state.identities = (identity("alice-token", "team", "alice"), identity("rotated-token", "team", "alice"), identity("bob-token", "team", "bob"), identity("tenant-b-token", "another", "alice"))
    assert client.get("/api/knowledge/libraries").status_code == 401
    client.headers["Authorization"] = "Bearer alice-token"
    base = library(client)
    client.headers["Authorization"] = "Bearer rotated-token"
    assert client.get(base + "/documents").status_code == 200
    for token in ["bob-token", "tenant-b-token"]:
        client.headers["Authorization"] = "Bearer " + token
        assert client.get(base + "/documents").status_code == 404


def test_storage_survives_restart_and_jobs_resume(tmp_path):
    service = KnowledgeService(tmp_path, start_worker=False)
    owner, token = service.store.session(None, create=True)
    lib = service.store.create_library(owner, "persistent")["id"]
    doc = service.store.put_document(owner, lib, "a.txt", b"Persistent Orchid policy")
    job = service.store.claim()
    assert job["id"] == doc["id"]
    with pytest.raises(RuntimeError, match="worker"):
        KnowledgeService(tmp_path)
    service.close()
    restarted = KnowledgeService(tmp_path)
    try:
        assert restarted.store.session(token)[0] == owner
        for _ in range(100):
            result = restarted.store.documents(owner, lib)[0]
            if result["status"] in {"ready", "failed"}:
                break
            time.sleep(.05)
        assert result["status"] == "ready", result
        assert restarted.search(owner, lib, "Orchid")
    finally:
        restarted.close()


def test_old_worker_cannot_resurrect_deleted_or_replaced_document(tmp_path):
    store = KnowledgeStore(tmp_path / "store.sqlite")
    lib = store.create_library("alice", "one")["id"]
    result = {"chunks": [{"ordinal": 1, "location": "line 1", "text": "obsolete"}], "warnings": []}
    doc = store.put_document("alice", lib, "a.txt", b"old")["id"]
    store.put_document("alice", lib, "a.txt", b"new", doc)
    store.finish(doc, 1, result)
    assert store.corpus("alice", lib) == []
    store.mutate_document("alice", lib, doc, "delete")
    store.finish(doc, 2, result)
    assert store.documents("alice", lib) == []


def test_docx_extract_preserves_paragraph_locations_and_ignores_external_links(tmp_path):
    path = tmp_path / "example.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Orchid 项目负责人是陈老师。</w:t></w:r></w:p></w:body></w:document>')
        archive.writestr("word/_rels/document.xml.rels", '<Relationships><Relationship Target="https://example.invalid/private" TargetMode="External"/></Relationships>')
    parsed = extract(path, ".docx")
    assert parsed["chunks"][0]["location"] == "第 1 段"
    assert "陈老师" in parsed["chunks"][0]["text"]


def test_docx_entity_and_zip_bomb_rejected(tmp_path):
    path = tmp_path / "bad.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", '<!DOCTYPE a [<!ENTITY x SYSTEM "file:///private">]><a>&x;</a>')
    with pytest.raises(Exception):
        extract(path, ".docx")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "a" * 100000)
    with pytest.raises(ValueError, match="压缩比"):
        extract(path, ".docx")


def pdf_bytes(text=None):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=500, height=500)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 30 400 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_real_pdf_upload_and_page_location(client):
    base = library(client)
    doc = upload(client, base, "research.pdf", pdf_bytes("Orchid archive deadline is Friday"))
    ready = wait_document(client, base, doc)
    assert ready["chunks"][0]["location"] == "第 1 页"
    assert "Friday" in client.post(base + "/ask", json={"question": "Orchid deadline"}).json()["answer"]


def test_blank_pdf_does_not_claim_success(client):
    base = library(client)
    doc = upload(client, base, "scan.pdf", pdf_bytes())
    assert "OCR" in wait_document(client, base, doc, "failed")["error"]


def test_utf8_and_chunk_limits(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes(b"\xff\x00")
    with pytest.raises(ValueError, match="UTF-8"):
        extract(path, ".txt")
    path.write_text("x" * 500001, encoding="utf-8")
    with pytest.raises(ValueError, match="50 万"):
        extract(path, ".txt")
    path.write_text("证据" * 600, encoding="utf-8")
    result = extract(path, ".txt")
    assert len(result["chunks"]) == 2
    assert all(len(c["text"]) <= 900 for c in result["chunks"])


def test_library_quota_and_no_empty_names(client):
    assert client.post("/api/knowledge/libraries", json={"name": " "}).status_code == 422
    for i in range(20):
        library(client, str(i))
    assert client.post("/api/knowledge/libraries", json={"name": "21"}).status_code == 409


def test_workspace_startup_does_not_need_enterprise_services(tmp_path, monkeypatch):
    import hashlib
    identities = tmp_path / "server-identities.json"
    identities.write_text(json.dumps([{"token_sha256": hashlib.sha256(b"workspace-test").hexdigest(),
        "principal": {"tenant_id": "team", "subject_id": "alice", "roles": ["employee"], "groups": [], "authz_version": "v1"},
        "retrieval_policy": {"policy_version": "v1", "allowed_sources": ["github"]}}]))
    monkeypatch.setenv("AGENT_MODE", "workspace")
    monkeypatch.setenv("AGENT_IDENTITY_FILE", str(identities))
    monkeypatch.setenv("AGENT_KNOWLEDGE_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("AGENT_KNOWLEDGE_MODEL_ENABLED", raising=False)
    with TestClient(app) as session:
        assert session.get("/api/config").json()["mode"] == "workspace"
        assert session.get("/health/ready").status_code == 401
        assert session.get("/health/ready", headers={"Authorization": "Bearer workspace-test"}).status_code == 200
        assert session.post("/ask", json={"question": "x"}, headers={"Authorization": "Bearer workspace-test"}).status_code == 409


def test_backup_roundtrip_does_not_overwrite_and_keeps_private_scope(tmp_path):
    from enterprise_agent.cli.knowledge_backup import backup
    store = KnowledgeStore(tmp_path / "source.sqlite3")
    owner, cookie = store.session(None, create=True)
    library_id = store.create_library(owner, "backup sample")["id"]
    doc = store.put_document(owner, library_id, "a.txt", b"Orchid backup test")
    destination = tmp_path / "backup.sqlite3"
    backup(store.path, destination)
    with pytest.raises(FileExistsError):
        backup(store.path, destination)
    restored = KnowledgeStore(destination)
    assert restored.session(cookie)[0] == owner
    assert restored.documents(owner, library_id)[0]["id"] == doc["id"]
    assert restored.libraries("another-owner") == []


def test_unknown_model_usage_not_reported_as_zero(client, monkeypatch):
    base = library(client)
    doc = upload(client, base)
    wait_document(client, base, doc)
    app.state.knowledge.model_settings = object()
    monkeypatch.setattr(app.state.knowledge, "_generate", lambda *args: SimpleNamespace(
        text='{"answer":"预算 [E1]","citations":["E1"]}', input_tokens=None, output_tokens=None))
    assert client.post(base + "/ask", json={"question": "Orchid", "answer_mode": "model"}).json()["model_token_count"] is None
