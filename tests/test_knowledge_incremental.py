"""Cost/resource invariants use synthetic vectors and temporary stores only."""

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.knowledge.profiles import Provider
from enterprise_agent.knowledge.resources import KnowledgeLimits, vector_reservation
from enterprise_agent.knowledge.retrieval import corpus_hash
from enterprise_agent.knowledge.service import KnowledgeService
from enterprise_agent.knowledge.store import KnowledgeStore, KnowledgeError
from enterprise_agent.web.api import app


def provider():
    return Provider(base_url="http://127.0.0.1:19999/v1", model="synthetic", dimensions=2)


def insert(store, library, name="a.txt", count=1, owner="alice"):
    doc = store.put_document(owner, library, name, f"synthetic {name}".encode())
    store.finish(doc["id"], 1, {"warnings": [], "chunks": [
        dict(ordinal=i+1, location=f"line {i+1}", text=f"alpha synthetic {name} {i}") for i in range(count)]})
    return doc["id"]


@pytest.fixture
def service(tmp_path):
    result = KnowledgeService(tmp_path, start_worker=False)
    result.update_profile("embedding", provider(), 0)
    yield result
    result.close()


@pytest.fixture
def network(monkeypatch):
    state = SimpleNamespace(batches=[], fail_at=None, during_call=None)
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def embed_texts(self, texts):
            state.batches.append(list(texts))
            if state.during_call:
                state.during_call()
            if len(state.batches) == state.fail_at:
                raise RuntimeError("synthetic failure; never expose provider body")
            return [[1.,0.] for _ in texts]
        def close(self):
            pass
    monkeypatch.setattr("enterprise_agent.infrastructure.embedding_client.OpenAICompatibleEmbeddingClient", Client)
    return state


def build(service, library, **kwargs):
    queued = service.store.queue_vectors("alice", library, provider().fingerprint, 2, **kwargs)
    job = service.store.claim_vectors()
    service.build_vectors(job)
    return queued, service.store.vector_status("alice", library, provider().fingerprint)


def test_incremental_add_delete_and_noop_make_no_duplicate_calls(service, network):
    s = service.store
    library = s.create_library("alice","library")["id"]
    first = insert(s,library,count=3)
    _, status = build(service,library)
    assert status["status"] == "ready" and status["embedded"] == 3
    plan = s.vector_plan("alice",library,provider().fingerprint,2)
    assert (plan["reused"],plan["pending"],plan["estimated_batches"],plan["characters_to_send"]) == (3,0,0,0)
    _, status = build(service,library,plan_id=plan["plan_id"])
    assert status["reused"] == 3 and len(network.batches) == 1
    insert(s,library,"new.txt",2)
    queued, status = build(service,library)
    assert queued["pending"] == 2 and status["reused"] == 3
    assert [len(batch) for batch in network.batches] == [3,2]
    s.mutate_document("alice",library,first,"delete")
    _, status = build(service,library)
    assert status["status"] == "ready" and status["reused"] == 2 and len(network.batches) == 2


def test_failed_batch_resume_persists_across_restart(service, network):
    s=service.store
    library=s.create_library("alice","library")["id"]
    insert(s,library,count=20)
    network.fail_at=2
    _, failed=build(service,library)
    assert failed["status"] == "failed" and failed["done"] == 16 and failed["reserved_bytes"] == 0
    with pytest.raises(KnowledgeError):
        s.read_vectors("alice",library,provider().fingerprint,corpus_hash(s.corpus("alice",library)))
    s.recover()
    reopened=KnowledgeStore(s.path)
    plan=reopened.vector_plan("alice",library,provider().fingerprint,2)
    assert plan["reused"] == 16 and plan["pending"] == 4
    network.fail_at=None
    _,ready=build(service,library,plan_id=plan["plan_id"])
    assert ready["status"] == "ready" and ready["reused"] == 16 and ready["embedded"] == 4
    assert [len(b) for b in network.batches] == [16,4,4]  # Uncertain failed calls can be billed again.


def test_forced_rebuild_reembeds_same_inputs(service,network):
    library=service.store.create_library("alice","library")["id"]
    insert(service.store,library)
    build(service,library)
    plan=service.store.vector_plan("alice",library,provider().fingerprint,2,force=True)
    assert plan["pending"] == 1 and plan["reused"] == 0
    _,status=build(service,library,force=True,plan_id=plan["plan_id"])
    assert status["embedded"] == 1 and len(network.batches) == 2


def test_plan_change_requires_reconfirmation(service,network):
    library=service.store.create_library("alice","library")["id"]
    insert(service.store,library)
    plan=service.store.vector_plan("alice",library,provider().fingerprint,2)
    insert(service.store,library,"b.txt")
    with pytest.raises(KnowledgeError) as failure:
        build(service,library,plan_id=plan["plan_id"])
    assert failure.value.status == 409 and network.batches == []


def test_partial_cache_never_shared_across_libraries_or_owners(service,network):
    first=service.store.create_library("alice","first")["id"]
    insert(service.store,first)
    build(service,first)
    for owner in ("alice","bob"):
        other=service.store.create_library(owner,"other")["id"]
        insert(service.store,other,owner=owner)
        assert service.store.vector_plan(owner,other,provider().fingerprint,2)["reused"] == 0
    with pytest.raises(KnowledgeError) as failure:
        service.store.vector_plan("bob",first,provider().fingerprint,2)
    assert failure.value.status == 404


def test_same_bytes_replacement_keeps_revision_and_vectors(service,network):
    s=service.store
    library=s.create_library("alice","library")["id"]
    doc=insert(s,library)
    build(service,library)
    result=s.put_document("alice",library,"a.txt",b"synthetic a.txt",doc)
    assert result["unchanged"] and result["revision"] == 1 and result["status"] == "ready"
    assert s.vector_status("alice",library,provider().fingerprint)["status"] == "ready"
    changed=s.put_document("alice",library,"a.txt",b"genuinely changed",doc)
    assert changed["revision"] == 2
    with s.db() as db:
        assert db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0


def test_quota_reservations_prevent_concurrent_overcommit(tmp_path):
    unit=vector_reservation(2)
    s=KnowledgeStore(tmp_path/"k.sqlite3",limits=KnowledgeLimits(vector_bytes=unit,owner_vector_bytes=unit))
    libraries=[s.create_library("alice",f"lib{i}")["id"] for i in range(2)]
    for library in libraries:
        insert(s,library)
    s.queue_vectors("alice",libraries[0],"profile",2)
    with pytest.raises(KnowledgeError) as failure:
        s.queue_vectors("alice",libraries[1],"profile",2)
    assert failure.value.status == 413
    assert s.storage_usage("alice")["vector_reserved_bytes"] == unit
    assert s.storage_usage("bob")["vector_reserved_bytes"] == 0
    s.cancel_vectors("alice",libraries[0])
    assert s.queue_vectors("alice",libraries[1],"profile",2)["status"] == "queued"


def test_owner_quota_distinct_from_global_quota(tmp_path):
    s=KnowledgeStore(tmp_path/"k.sqlite3",limits=KnowledgeLimits(vector_bytes=1024,owner_vector_bytes=vector_reservation(2)))
    a=s.create_library("alice","a")["id"]
    b=s.create_library("bob","b")["id"]
    insert(s,a,count=2)
    insert(s,b,owner="bob")
    with pytest.raises(KnowledgeError):
        s.queue_vectors("alice",a,"profile",2)
    assert s.queue_vectors("bob",b,"profile",2)["status"] == "queued"


def test_simultaneous_queue_reservations_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    unit=vector_reservation(2)
    s=KnowledgeStore(tmp_path/"k.sqlite3",limits=KnowledgeLimits(vector_bytes=unit,owner_vector_bytes=unit))
    libraries=[s.create_library("alice",f"lib{i}")["id"] for i in range(2)]
    for library in libraries:
        insert(s,library)
    barrier=threading.Barrier(2)
    def enqueue(library):
        barrier.wait(timeout=3)
        try:
            s.queue_vectors("alice",library,"profile",2)
            return 202
        except KnowledgeError as error:
            return error.status
    with ThreadPoolExecutor(max_workers=2) as workers:
        statuses=list(workers.map(enqueue,libraries))
    assert sorted(statuses)==[202,413]
    assert s.storage_usage("alice")["vector_reserved_bytes"]==unit


def test_low_disk_prevents_upload_and_paid_index(service,network,monkeypatch):
    s=service.store
    library=s.create_library("alice","a")["id"]
    insert(s,library)
    monkeypatch.setattr("enterprise_agent.knowledge.store.shutil.disk_usage",lambda path: SimpleNamespace(free=1))
    with pytest.raises(KnowledgeError) as failure:
        s.put_document("alice",library,"b.txt",b"new")
    assert failure.value.status == 507
    with pytest.raises(KnowledgeError) as failure:
        build(service,library)
    assert failure.value.status == 507 and network.batches == []


def test_cancel_restart_old_worker_cannot_write_new_job(service,network):
    s=service.store
    library=s.create_library("alice","a")["id"]
    insert(s,library)
    s.queue_vectors("alice",library,provider().fingerprint,2)
    old=s.claim_vectors()
    batch=s.pending_vectors(old)
    s.cancel_vectors("alice",library)
    s.queue_vectors("alice",library,provider().fingerprint,2)
    new=s.claim_vectors()
    assert old["job_id"] != new["job_id"]
    with pytest.raises(KnowledgeError):
        s.save_vector_batch(old,batch,[[1,0]])
    s.fail_vectors(old,"old error")
    assert s.vector_status("alice",library,provider().fingerprint)["status"] == "building"
    service.build_vectors(new)
    assert s.vector_status("alice",library,provider().fingerprint)["status"] == "ready"


def test_clear_index_preserves_documents_and_rejects_inflight(service,network):
    s=service.store
    library=s.create_library("alice","a")["id"]
    insert(s,library)
    build(service,library)
    s.queue_vectors("alice",library,provider().fingerprint,2,force=True)
    job=s.claim_vectors()
    batch=s.pending_vectors(job)
    s.clear_vectors("alice",library)
    assert s.storage_usage("alice")["vector_bytes"] == 0
    assert len(s.corpus("alice",library)) == 1
    with pytest.raises(KnowledgeError):
        s.save_vector_batch(job,batch,[[1,0]])
    assert s.vector_status("alice",library,provider().fingerprint)["status"] == "not_built"


def test_file_change_while_network_waits_discards_batch(service,network):
    s=service.store
    library=s.create_library("alice","a")["id"]
    doc=insert(s,library)
    network.during_call=lambda:s.mutate_document("alice",library,doc,"delete")
    _,status=build(service,library)
    assert status["status"] == "failed" and status["reserved_bytes"] == 0
    assert s.storage_usage("alice")["vector_bytes"] == 0


def test_v2_vectors_migrate_without_reembedding(service,network):
    s=service.store
    library=s.create_library("alice","a")["id"]
    insert(s,library)
    build(service,library)
    with s.db(True) as db:
        for field in ("profile_hash","input_hash","dimensions"):
            db.execute(f"ALTER TABLE vectors DROP COLUMN {field}")
        for field in ("job_id","dimensions","reused","embedded","reserved_bytes"):
            db.execute(f"ALTER TABLE vector_jobs DROP COLUMN {field}")
        db.execute("PRAGMA user_version=2")
    migrated=KnowledgeStore(s.path)
    assert migrated.vector_plan("alice",library,provider().fingerprint,2)["reused"] == 1
    backups=list(s.path.parent.glob("before-v3-*.sqlite3"))
    assert len(backups)==1
    with sqlite3.connect(backups[0]) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_migration_error_rolls_back_schema_and_keeps_backup(service,network,monkeypatch):
    s=service.store
    library=s.create_library("alice","a")["id"]
    insert(s,library)
    build(service,library)
    with s.db(True) as db:
        for field in ("profile_hash","input_hash","dimensions"):
            db.execute(f"ALTER TABLE vectors DROP COLUMN {field}")
        for field in ("job_id","dimensions","reused","embedded","reserved_bytes"):
            db.execute(f"ALTER TABLE vector_jobs DROP COLUMN {field}")
        db.execute("PRAGMA user_version=2")
    def interrupted(row):
        raise RuntimeError("synthetic migration failure")
    monkeypatch.setattr("enterprise_agent.knowledge.retrieval.embedding_hash",interrupted)
    with pytest.raises(RuntimeError):
        KnowledgeStore(s.path)
    with s.db() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0]==2
        assert "input_hash" not in {row["name"] for row in db.execute("PRAGMA table_info(vectors)")}
        assert db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]==1
    assert len(list(s.path.parent.glob("before-v3-*.sqlite3")))==1


def test_parser_not_blocked_by_slow_vector_job(tmp_path,network):
    entered,release=threading.Event(),threading.Event()
    def delay():
        entered.set()
        assert release.wait(5)
    network.during_call=delay
    service=KnowledgeService(tmp_path)
    try:
        service.update_profile("embedding",provider(),0)
        s=service.store
        library=s.create_library("alice","a")["id"]
        insert(s,library)
        s.queue_vectors("alice",library,provider().fingerprint,2)
        service.wake.set()
        assert entered.wait(3)
        fresh=s.put_document("alice",library,"fresh.txt",b"newly uploaded text")
        service.wake.set()
        end=time.monotonic()+3
        while time.monotonic()<end:
            if s.preview("alice",library,fresh["id"])["status"] == "ready":
                break
            time.sleep(.02)
        assert s.preview("alice",library,fresh["id"])["status"] == "ready"
    finally:
        release.set()
        service.close()


def test_api_estimate_consent_clear_and_identity(monkeypatch,tmp_path,network):
    monkeypatch.setenv("AGENT_MODE","demo")
    monkeypatch.setenv("AGENT_KNOWLEDGE_DIR",str(tmp_path))
    monkeypatch.delenv("AGENT_KNOWLEDGE_MODEL_ENABLED",raising=False)
    with TestClient(app,base_url="http://127.0.0.1",client=("127.0.0.1",1234)) as c:
        c.get("/api/knowledge/session")
        service=app.state.knowledge
        service.update_profile("embedding",provider(),0)
        owner=service.store.session(c.cookies.get("ea_knowledge_session"))[0]
        library=service.store.create_library(owner,"a")["id"]
        insert(service.store,library,owner=owner)
        base=f"/api/knowledge/libraries/{library}"
        plan=c.get(base+"/index/plan").json()
        assert plan["pending"]==1 and network.batches==[]
        assert c.post(base+"/index",json={"consent":False,"plan_id":plan["plan_id"]}).status_code==422
        assert c.post(base+"/index",json={"consent":True,"plan_id":"0"*64}).status_code==409
        assert c.post(base+"/index",json={"consent":True,"plan_id":plan["plan_id"]}).status_code==202
        assert c.get("/api/knowledge/storage").json()["documents"]==1
        other=TestClient(app,base_url="http://127.0.0.1",client=("127.0.0.1",1235))
        try:
            other.get("/api/knowledge/session")
            assert other.get(base+"/index/plan").status_code==404
            assert other.delete(base+"/vectors").status_code==404
            assert other.get("/api/knowledge/storage").json()["documents"]==0
        finally:
            other.close()
        assert c.delete(base+"/vectors").status_code==204
        assert c.get(base+"/documents").json()[0]["status"]=="ready"
        assert c.get(base+"/index").json()["status"]=="not_built"


@pytest.mark.parametrize("value", ["nope", "0", "-1"])
def test_invalid_resource_environment_rejected(monkeypatch,value):
    monkeypatch.setenv("AGENT_VECTOR_BYTES_LIMIT",value)
    with pytest.raises(RuntimeError):
        KnowledgeLimits.from_environment()
