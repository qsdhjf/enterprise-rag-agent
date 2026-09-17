import json

import httpx
import pytest

from enterprise_agent.ingestion.connectors import ConnectorACL, export_chunks, folder_documents, github_issue_documents


def test_folder_export_has_stable_ids_and_explicit_acl(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "runbook.md").write_text("# Release\nAlways verify the build.", encoding="utf-8")
    (root / ".env").write_text("do-not-import", encoding="utf-8")
    acl = ConnectorACL("team-a", ("group:engineering",))
    output = tmp_path / "chunks.jsonl"
    assert export_chunks(folder_documents(root, "confluence", acl), output, "team-a") == 1
    chunk = json.loads(output.read_text(encoding="utf-8"))
    assert chunk["tenant_id"] == "team-a"
    assert chunk["visibility"] == "restricted"
    assert chunk["acl_principals"] == ["group:engineering"]
    assert chunk["chunk_id"].startswith("chunk_v2_")
    with pytest.raises(FileExistsError):
        export_chunks([], output, "team-a")


def test_failed_export_does_not_publish_partial_file(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("valid", encoding="utf-8")
    (root / "b.txt").write_bytes(b"\xff\xff")
    output = tmp_path / "chunks.jsonl"
    with pytest.raises(UnicodeDecodeError):
        export_chunks(folder_documents(root, "confluence", ConnectorACL("a", ("user:a",))), output, "a")
    assert not output.exists()
    assert not list(tmp_path.glob(".export-*"))


def test_folder_size_limit_aborts_instead_of_truncating(tmp_path):
    (tmp_path / "a.txt").write_text("too big", encoding="utf-8")
    with pytest.raises(ValueError, match="limit"):
        list(folder_documents(tmp_path, "confluence", ConnectorACL("a", ("user:a",)), max_bytes=2))


def test_github_connector_is_fixed_host_read_only_and_skips_prs():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=[
            {"number": 7, "title": "Bug", "body": "Steps", "created_at": "2026-01-01"},
            {"number": 8, "title": "PR", "pull_request": {"url": "ignored"}},
        ])
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        docs = list(github_issue_documents("example/project", ConnectorACL("a", ("user:a",)), client=client))
    assert len(docs) == 1
    assert docs[0].document_id == "github:example/project:issue:7"
    assert requests[0].method == "GET" and requests[0].url.host == "api.github.com"
    assert requests[0].url.params["state"] == "all"
    with pytest.raises(ValueError, match="owner/name"):
        list(github_issue_documents("http://127.0.0.1/private", ConnectorACL("a", ("user:a",))))


def test_github_pagination_failure_leaves_no_export(tmp_path):
    def handle(request):
        if request.url.params["page"] == "1":
            return httpx.Response(200, json=[{"number": n, "title": "Issue", "body": "text"} for n in range(100)])
        return httpx.Response(403, json={"message": "rate limited"})
    output = tmp_path / "chunks.jsonl"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            export_chunks(github_issue_documents("a/b", ConnectorACL("a", ("user:a",)), client=client), output, "a")
    assert not output.exists()


def test_connector_rejects_empty_acl():
    with pytest.raises(RuntimeError, match="ACL_PRINCIPALS_REQUIRED"):
        ConnectorACL("a", ()).fields()
