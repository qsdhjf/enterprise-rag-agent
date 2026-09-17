"""Opt-in test against a DISPOSABLE cluster; touches only a UUID-owned namespace."""

import json
import os
from uuid import uuid4

import pytest
from opensearchpy import OpenSearch

from enterprise_agent.ingestion.opensearch.snapshot import publish_snapshot
from enterprise_agent.retrieval.access_control import RetrievalAccessContext
from enterprise_agent.retrieval.keyword_search import search_chunks
from enterprise_agent.retrieval.vector_search import vector_search_chunks


@pytest.mark.skipif(not os.environ.get("EA_TEST_OPENSEARCH_URL"), reason="Set EA_TEST_OPENSEARCH_URL to a disposable cluster")
@pytest.mark.parametrize("hybrid", [False, True])
def test_real_opensearch_snapshot_replacement_and_alias_read(tmp_path, hybrid):
    client = OpenSearch(hosts=[os.environ["EA_TEST_OPENSEARCH_URL"]])
    prefix = "ea-ci-" + uuid4().hex
    alias = prefix + "-current"
    tenant = prefix
    row = dict(chunk_id="old-untrusted-id", document_id="doc", source="jira", title="Archive policy",
               sequence=0, text="Archive project carefully", created_at=None, updated_at=None,
               original_location=None, tenant_id=tenant, visibility="restricted",
               acl_principals=["user:alice"], acl_version="v1")
    path = tmp_path / "chunks.jsonl"
    class TestEmbedding:
        def embed_text(self, text):
            return [0.1, 0.2, 0.3]
        def embed_texts(self, texts):
            return [self.embed_text(text) for text in texts]
    embedding = TestEmbedding()
    options = (dict(embedding_client=embedding, embedding_model="integration-test-only", dimensions=3)
               if hybrid else {})
    try:
        path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "document_id": "deleted"}) + "\n", encoding="utf-8")
        first = publish_snapshot(client, alias, path, tenant, confirm_full_snapshot=True, **options)
        assert client.count(index=alias)["count"] == 2
        path.write_text(json.dumps({**row, "text": "Revised archive procedure"}) + "\n", encoding="utf-8")
        second = publish_snapshot(client, alias, path, tenant, confirm_full_snapshot=True, **options)
        assert second.previous_index == first.index
        assert client.count(index=alias)["count"] == 1
        assert client.count(index=first.index)["count"] == 2
        access = RetrievalAccessContext(tenant, "alice", frozenset({"employee"}), frozenset(), "v1")
        hits = search_chunks(client, alias, "archive", access_context=access)
        assert len(hits) == 1 and hits[0].text == "Revised archive procedure"
        assert client.get(index=alias, id=hits[0].chunk_id)["_source"]["document_id"] == "doc"
        other = RetrievalAccessContext("different-tenant", "alice", frozenset({"employee"}), frozenset(), "v1")
        assert search_chunks(client, alias, "archive", access_context=other) == []
        if hybrid:
            assert vector_search_chunks(client, alias, embedding, "archive", access_context=access)[0].document_id == "doc"
            assert vector_search_chunks(client, alias, embedding, "archive", access_context=other) == []
    finally:
        # UUID prefix is created by THIS test; never enumerate/delete a user index.
        mappings = client.indices.get_mapping(index=alias + "--*", allow_no_indices=True)
        for index in mappings:
            assert index.startswith(alias + "--") and len(index) > len(alias) + 2
            client.indices.delete(index=index)
        client.close()
