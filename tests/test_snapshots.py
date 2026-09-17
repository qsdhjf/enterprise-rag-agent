"""Index lifecycle guarantees without network or modifying an existing index."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from enterprise_agent.ingestion.data_reader import EnterpriseDocument
from enterprise_agent.ingestion.chunker import chunk_document
from enterprise_agent.ingestion.identity import normalize_chunk_identity
from enterprise_agent.ingestion.opensearch.ingestion import BulkIndexResult, iter_index_actions
import enterprise_agent.ingestion.opensearch.snapshot as module


def record(**changes):
    value = dict(chunk_id="legacy-same", document_id="doc", source="jira", title="Title", sequence=0,
                 text="old text", tenant_id="tenant-a", visibility="restricted", acl_principals=["user:alice"],
                 acl_version="v1", created_at=None, updated_at=None, original_location=None)
    value.update(changes)
    return value


def jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class Indexes:
    def __init__(self):
        self.bodies = {}
        self.docs = {}
        self.aliases = {}
        self.blocks = set()
        self.switches = []
    def exists_alias(self, name): return name in self.aliases
    def exists(self, index): return index in self.bodies
    def get_alias(self, name): return {self.aliases[name]: {"aliases": {name: {}}}}
    def get_mapping(self, index): return {index: {"mappings": self.bodies[index]["mappings"]}}
    def create(self, index, body):
        assert index not in self.bodies
        self.bodies[index] = body
        self.docs[index] = {}
    def put_settings(self, index, body):
        assert body == {"index.blocks.write": True}
        self.blocks.add(index)
    def update_aliases(self, body):
        self.switches.append(body)
        for action in body["actions"]:
            if "remove" in action:
                removal = action["remove"]
                assert removal["must_exist"] is True
                assert self.aliases.pop(removal["alias"]) == removal["index"]
            else:
                addition = action["add"]
                assert addition["index"] in self.blocks
                assert addition["is_write_index"] is False
                self.aliases[addition["alias"]] = addition["index"]
        return {"acknowledged": True}


class Client:
    def __init__(self): self.indices = Indexes()
    def count(self, index): return {"count": len(self.indices.docs[index])}


@pytest.fixture
def client(monkeypatch):
    def bulk(client, index_name, jsonl_path, batch_size, expected_tenant_id):
        actions = list(iter_index_actions(jsonl_path, index_name, expected_tenant_id=expected_tenant_id))
        client.indices.docs[index_name] = {item["_id"]: item["_source"] for item in actions}
        return BulkIndexResult(len(actions), 0)
    monkeypatch.setattr(module, "bulk_index_jsonl", bulk)
    return Client()


def publish(client, path, **options):
    return module.publish_snapshot(client, "tenant-a-current", path, "tenant-a", confirm_full_snapshot=True, **options)


def test_id_namespaces_are_deterministic_and_legacy_ids_are_not_trusted(tmp_path):
    first = normalize_chunk_identity(record())
    assert first == normalize_chunk_identity(record())
    assert first["chunk_id"] != normalize_chunk_identity(record(tenant_id="tenant-b"))["chunk_id"]
    assert first["chunk_id"] != normalize_chunk_identity(record(source="slack"))["chunk_id"]
    path = jsonl(tmp_path / "rows.jsonl", [record(), record(tenant_id="tenant-b")])
    actions = list(iter_index_actions(path, "index"))
    assert actions[0]["_id"] != actions[1]["_id"]
    assert all(action["_id"] == action["_source"]["chunk_id"] for action in actions)
    doc = EnterpriseDocument("doc", "jira", "Title", "same", None, None, None, "a")
    assert chunk_document(doc)[0].chunk_id != chunk_document(replace(doc, tenant_id="b"))[0].chunk_id


def test_replacement_removes_stale_chunks_and_deleted_documents_but_retains_old_generation(client, tmp_path):
    first = publish(client, jsonl(tmp_path / "a.jsonl", [record(), record(document_id="deleted")]))
    second = publish(client, jsonl(tmp_path / "b.jsonl", [record(text="new text")]))
    assert client.indices.aliases[second.alias] == second.index
    assert second.previous_index == first.index
    assert len(client.indices.docs[first.index]) == 2
    assert [doc["text"] for doc in client.indices.docs[second.index].values()] == ["new text"]
    assert all(doc["document_id"] != "deleted" for doc in client.indices.docs[second.index].values())


@pytest.mark.parametrize("failure", ["partial", "exception", "count"])
def test_failed_stage_does_not_switch_live_alias(client, tmp_path, monkeypatch, failure):
    path = jsonl(tmp_path / "a.jsonl", [record()])
    first = publish(client, path)
    def broken(*args, **kwargs):
        if failure == "exception": raise RuntimeError("unavailable")
        return BulkIndexResult(0, 1) if failure == "partial" else BulkIndexResult(1, 0)
    monkeypatch.setattr(module, "bulk_index_jsonl", broken)
    with pytest.raises(RuntimeError, match="Snapshot incomplete"):
        publish(client, path)
    assert client.indices.aliases[first.alias] == first.index
    assert len(client.indices.switches) == 1


@pytest.mark.parametrize("rows", [[], [record(), record(text="conflicting version")], [record(tenant_id="other")]])
def test_invalid_snapshot_rejected_before_index_creation(client, tmp_path, rows):
    with pytest.raises((ValueError, RuntimeError)):
        publish(client, jsonl(tmp_path / "a.jsonl", rows))
    assert not client.indices.bodies


def test_empty_snapshot_requires_explicit_intent(client, tmp_path):
    result = publish(client, jsonl(tmp_path / "empty.jsonl", []), allow_empty=True)
    assert result.chunks == 0


def test_managed_alias_cannot_be_replaced_by_another_tenant(client, tmp_path):
    result = publish(client, jsonl(tmp_path / "a.jsonl", [record()]))
    path = jsonl(tmp_path / "b.jsonl", [record(tenant_id="tenant-b")])
    with pytest.raises(ValueError, match="different-tenant"):
        module.publish_snapshot(client, result.alias, path, "tenant-b", confirm_full_snapshot=True)
    assert client.indices.aliases[result.alias] == result.index


def test_physical_index_is_never_replaced_or_deleted(client, tmp_path):
    client.indices.create("tenant-a-current", {"mappings": {}})
    with pytest.raises(ValueError, match="physical"):
        publish(client, jsonl(tmp_path / "a.jsonl", [record()]))
    assert not client.indices.switches
