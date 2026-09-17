"""验证摄取链路严格传播 ACL，并对旧本地样本做显式、窄范围兼容。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from enterprise_agent.ingestion.chunker import DocumentChunk, chunk_document
from enterprise_agent.ingestion.data_reader import (
    EnterpriseDocument,
    load_document,
    load_local_demo_document,
)
from enterprise_agent.ingestion.jsonl_writer import write_chunks_jsonl
from enterprise_agent.ingestion.opensearch.hybrid_ingestion import (
    bulk_index_hybrid_jsonl,
    iter_chunk_batches,
)
from enterprise_agent.ingestion.opensearch.index import (
    CHUNK_INDEX_BODY,
    ensure_chunk_index,
)
from enterprise_agent.ingestion.opensearch.ingestion import iter_index_actions
from enterprise_agent.retrieval.access_control import (
    LOCAL_DEMO_ACL_VERSION,
    LOCAL_DEMO_TENANT_ID,
    RetrievalAuthorizationError,
    RetrievalAccessContext,
    authorize_document_metadata,
)
from enterprise_agent.retrieval.keyword_search import search_chunks


def _source_document(**overrides: Any) -> dict[str, Any]:
    """构造可按需覆盖字段的合法源文档。"""

    document: dict[str, Any] = {
        "dataset_doc_uuid": "doc-1",
        "title_field_name": "name",
        "content_field_names": ["body"],
        "name": "安全手册",
        "body": "只有授权主体可以读取这段内容。",
        "tenant_id": "tenant-a",
        "visibility": "restricted",
        "acl_principals": ["user:alice", "group:engineering"],
        "acl_version": "acl-v7",
    }
    document.update(overrides)
    return document


def _write_json(path: Path, value: object) -> Path:
    """把测试值写入 JSON 文件并返回路径。"""

    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _chunk_record(**overrides: Any) -> dict[str, Any]:
    """构造可按需覆盖 ACL 字段的合法分块记录。"""

    record: dict[str, Any] = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "source": "wiki",
        "title": "安全手册",
        "sequence": 0,
        "text": "授权正文",
        "created_at": None,
        "updated_at": None,
        "original_location": None,
        "tenant_id": "tenant-a",
        "visibility": "restricted",
        "acl_principals": ["user:alice"],
        "acl_version": "acl-v7",
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, *records: dict[str, Any]) -> Path:
    """把测试记录写入 JSONL 文件并返回路径。"""

    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_acl_survives_reader_chunker_and_jsonl_writer(tmp_path: Path) -> None:
    """验证读取、分块和 JSONL 写出全程保留 ACL。"""

    source_path = _write_json(tmp_path / "source.json", _source_document())

    document = load_document(
        source_path,
        "wiki",
        expected_tenant_id="tenant-a",
    )
    assert document.tenant_id == "tenant-a"
    assert document.visibility == "restricted"
    assert set(document.acl_principals) == {"user:alice", "group:engineering"}
    assert document.acl_version == "acl-v7"

    chunks = chunk_document(document, max_chars=100, overlap_chars=0)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.tenant_id == document.tenant_id
    assert chunk.visibility == document.visibility
    assert chunk.acl_principals == document.acl_principals
    assert chunk.acl_version == document.acl_version

    output_path = tmp_path / "chunks.jsonl"
    assert write_chunks_jsonl(
        chunks,
        output_path,
        expected_tenant_id="tenant-a",
    ) == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["tenant_id"] == "tenant-a"
    assert record["visibility"] == "restricted"
    assert set(record["acl_principals"]) == {"user:alice", "group:engineering"}
    assert record["acl_version"] == "acl-v7"


def test_reader_accepts_explicit_nested_acl_but_rejects_conflicts(
    tmp_path: Path,
) -> None:
    """验证读取器接受嵌套 ACL，并拒绝相互冲突的声明。"""

    nested_acl = {
        "tenant_id": "tenant-a",
        "visibility": "private",
        "acl_principals": ["role:auditor"],
        "acl_version": "acl-v3",
    }
    nested_path = _write_json(
        tmp_path / "nested.json",
        {
            "dataset_doc_uuid": "nested",
            "title_field_name": "name",
            "content_field_names": ["body"],
            "name": "审计记录",
            "body": "仅审计员可见",
            "metadata": {"access_control": nested_acl},
        },
    )
    document = load_document(
        nested_path,
        "wiki",
        expected_tenant_id="tenant-a",
    )
    assert document.tenant_id == "tenant-a"
    assert document.visibility == "private"
    assert document.acl_principals == ("role:auditor",)

    conflict_path = _write_json(
        tmp_path / "conflict.json",
        {
            **_source_document(),
            "access_control": {
                **nested_acl,
                "tenant_id": "tenant-b",
            },
        },
    )
    with pytest.raises((ValueError, RetrievalAuthorizationError), match="CONFLICT|冲突"):
        load_document(conflict_path, "wiki")


@pytest.mark.parametrize(
    ("changes", "error_code"),
    [
        ({"acl_version": None}, "ACL_VERSION_INVALID"),
        ({"visibility": "external"}, "VISIBILITY_INVALID"),
        ({"acl_principals": ["engineering"]}, "ACL_PRINCIPALS_INVALID"),
        ({"acl_principals": []}, "ACL_PRINCIPALS_REQUIRED"),
        ({"tenant_id": "", "visibility": "tenant"}, "TENANT_INVALID"),
    ],
)
def test_reader_rejects_malformed_acl(
    tmp_path: Path,
    changes: dict[str, Any],
    error_code: str,
) -> None:
    """参数化验证读取器以对应错误码拒绝各类畸形 ACL。"""

    source_path = _write_json(
        tmp_path / f"bad-{error_code}.json",
        _source_document(**changes),
    )
    with pytest.raises(RetrievalAuthorizationError, match=error_code):
        load_document(source_path, "wiki")


def test_reader_rejects_missing_and_cross_tenant_acl(tmp_path: Path) -> None:
    """验证读取器拒绝缺失 ACL 和跨租户文档。"""

    missing = _source_document()
    missing.pop("acl_version")
    missing_path = _write_json(tmp_path / "missing.json", missing)
    with pytest.raises(RetrievalAuthorizationError, match="ACL_METADATA_INCOMPLETE"):
        load_document(missing_path, "wiki")

    cross_tenant_path = _write_json(
        tmp_path / "cross-tenant.json",
        _source_document(tenant_id="tenant-b"),
    )
    with pytest.raises(RetrievalAuthorizationError, match="TENANT_MISMATCH"):
        load_document(
            cross_tenant_path,
            "wiki",
            expected_tenant_id="tenant-a",
        )


def test_local_demo_loader_is_explicit_and_rejects_explicit_production_acl(
    tmp_path: Path,
) -> None:
    """验证本地兼容需显式调用，且不能覆盖生产 ACL。"""

    # 旧生成数据中的 visibility 可能只是业务字段；显式 demo helper 才能覆盖它。
    legacy_path = _write_json(
        tmp_path / "legacy.json",
        {
            "dataset_doc_uuid": "legacy",
            "title_field_name": "name",
            "content_field_names": ["body"],
            "name": "旧样本",
            "body": "本地演示正文",
            "visibility": "company-only",
        },
    )
    with pytest.raises(RetrievalAuthorizationError):
        load_document(legacy_path, "wiki")

    document = load_local_demo_document(legacy_path, "wiki")
    assert document.tenant_id == LOCAL_DEMO_TENANT_ID
    assert document.visibility == "local-demo"
    assert document.acl_principals == ()
    assert document.acl_version == LOCAL_DEMO_ACL_VERSION

    production_path = _write_json(
        tmp_path / "production.json",
        _source_document(),
    )
    with pytest.raises(RetrievalAuthorizationError, match="TENANT_MISMATCH"):
        load_local_demo_document(production_path, "wiki")


def test_old_dataclass_constructors_remain_usable_but_writer_fails_closed(
    tmp_path: Path,
) -> None:
    """验证旧构造器仍可用，但缺失 ACL 的写出会关闭失败。"""

    document = EnterpriseDocument(
        document_id="legacy",
        source="wiki",
        title="旧构造器",
        content="正文",
        created_at=None,
        updated_at=None,
        original_location=None,
    )
    chunk = chunk_document(document, max_chars=100, overlap_chars=0)[0]
    assert isinstance(chunk, DocumentChunk)

    with pytest.raises(RetrievalAuthorizationError, match="ACL_METADATA_MISSING"):
        write_chunks_jsonl([chunk], tmp_path / "legacy.jsonl")


def test_keyword_ingestion_validates_acl_and_only_explicitly_enriches_legacy_local(
    tmp_path: Path,
) -> None:
    """验证关键词入库校验 ACL，且仅显式补全旧本地数据。"""

    valid_path = _write_jsonl(tmp_path / "valid.jsonl", _chunk_record())
    action = list(
        iter_index_actions(
            valid_path,
            "chunks",
            expected_tenant_id="tenant-a",
        )
    )[0]
    assert action["_source"]["tenant_id"] == "tenant-a"
    assert action["_source"]["acl_principals"] == ["user:alice"]
    assert action["_source"]["acl_version"] == "acl-v7"

    legacy = _chunk_record()
    for field in ("tenant_id", "visibility", "acl_principals", "acl_version"):
        legacy.pop(field)
    legacy_path = _write_jsonl(tmp_path / "legacy.jsonl", legacy)
    with pytest.raises(RetrievalAuthorizationError, match="ACL_METADATA_MISSING"):
        list(iter_index_actions(legacy_path, "chunks"))

    local_action = list(
        iter_index_actions(
            legacy_path,
            "chunks",
            allow_legacy_local=True,
        )
    )[0]
    assert local_action["_source"]["tenant_id"] == LOCAL_DEMO_TENANT_ID
    assert local_action["_source"]["visibility"] == "local-demo"
    assert local_action["_source"]["acl_principals"] == []
    assert local_action["_source"]["acl_version"] == LOCAL_DEMO_ACL_VERSION


def test_keyword_and_hybrid_ingestion_reject_cross_tenant_jsonl(
    tmp_path: Path,
) -> None:
    """验证关键词与混合入库都拒绝跨租户 JSONL。"""

    path = _write_jsonl(
        tmp_path / "cross.jsonl",
        _chunk_record(tenant_id="tenant-b"),
    )
    with pytest.raises(RetrievalAuthorizationError, match="TENANT_MISMATCH"):
        list(iter_index_actions(path, "chunks", expected_tenant_id="tenant-a"))
    with pytest.raises(RetrievalAuthorizationError, match="TENANT_MISMATCH"):
        list(iter_chunk_batches(path, 8, expected_tenant_id="tenant-a"))


def test_hybrid_ingestion_preserves_validated_acl_in_bulk_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证混合批量入库在最终 source 中保留已校验 ACL。"""

    path = _write_jsonl(tmp_path / "hybrid.jsonl", _chunk_record())
    captured_actions: list[dict[str, Any]] = []

    def fake_streaming_bulk(*, actions: list[dict[str, Any]], **_: Any):
        """记录批量动作并逐条返回成功结果。"""

        captured_actions.extend(actions)
        yield True, {}

    monkeypatch.setattr(
        "enterprise_agent.ingestion.opensearch.hybrid_ingestion.streaming_bulk",
        fake_streaming_bulk,
    )

    class FakeEmbeddingClient:
        """为每条测试文本返回固定向量。"""

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            """确认输入非空并返回固定二维向量。"""

            assert texts
            return [[0.1, 0.2] for _ in texts]

    class FakeIndices:
        """提供记录刷新目标的最小索引接口。"""

        def refresh(self, *, index: str) -> None:
            """验证批量入库刷新了目标索引。"""

            assert index == "chunks"

    class FakeOpenSearch:
        """暴露混合入库所需的最小索引客户端。"""

        indices = FakeIndices()

    result = bulk_index_hybrid_jsonl(
        client=FakeOpenSearch(),  # type: ignore[arg-type]
        index_name="chunks",
        jsonl_path=path,
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        expected_tenant_id="tenant-a",
    )
    assert (result.embedded, result.succeeded, result.failed) == (1, 1, 0)
    source = captured_actions[0]["_source"]
    assert source["tenant_id"] == "tenant-a"
    assert source["visibility"] == "restricted"
    assert source["acl_principals"] == ["user:alice"]
    assert source["acl_version"] == "acl-v7"


def test_hybrid_legacy_compatibility_requires_explicit_local_flag(
    tmp_path: Path,
) -> None:
    """验证混合入库仅在显式开关下兼容旧本地记录。"""

    legacy = _chunk_record()
    for field in ("tenant_id", "visibility", "acl_principals", "acl_version"):
        legacy.pop(field)
    path = _write_jsonl(tmp_path / "legacy-hybrid.jsonl", legacy)

    with pytest.raises(RetrievalAuthorizationError, match="ACL_METADATA_MISSING"):
        list(iter_chunk_batches(path, 8))

    record = list(
        iter_chunk_batches(path, 8, allow_legacy_local=True)
    )[0][0]
    assert record["tenant_id"] == LOCAL_DEMO_TENANT_ID
    assert record["visibility"] == "local-demo"
    assert record["acl_principals"] == []
    assert record["acl_version"] == LOCAL_DEMO_ACL_VERSION


def test_chunk_index_mapping_contains_the_complete_acl_contract() -> None:
    """验证分块索引映射包含完整 ACL keyword 字段。"""

    properties = CHUNK_INDEX_BODY["mappings"]["properties"]
    assert properties["tenant_id"]["type"] == "keyword"
    assert properties["visibility"]["type"] == "keyword"
    assert properties["acl_principals"]["type"] == "keyword"
    assert properties["acl_version"]["type"] == "keyword"


def test_local_legacy_action_round_trips_through_retrieval_authorization(
    tmp_path: Path,
) -> None:
    """旧 JSONL 显式补全后，本地主体可检索，生产主体仍被租户门禁拒绝。"""

    legacy = _chunk_record()
    for field in ("tenant_id", "visibility", "acl_principals", "acl_version"):
        legacy.pop(field)
    path = _write_jsonl(tmp_path / "round-trip.jsonl", legacy)
    source = list(
        iter_index_actions(
            path,
            "chunks",
            allow_legacy_local=True,
        )
    )[0]["_source"]

    class FakeSearch:
        """返回显式补全后的本地记录以验证检索闭环。"""

        def search(self, *, index: str, body: dict[str, Any]):
            """返回单条预设检索命中。"""

            return {
                "hits": {
                    "hits": [{"_score": 1.0, "_source": source}]
                }
            }

    local = RetrievalAccessContext.local_demo()
    assert authorize_document_metadata(source, local).tenant_id == (
        LOCAL_DEMO_TENANT_ID
    )
    hits = search_chunks(
        client=FakeSearch(),  # type: ignore[arg-type]
        index_name="chunks",
        query="授权正文",
        access_context=local,
    )
    assert hits[0].acl_version == LOCAL_DEMO_ACL_VERSION

    production = RetrievalAccessContext(
        tenant_id="tenant-a",
        subject_id="alice",
        roles=frozenset({"employee"}),
        groups=frozenset(),
        authz_version="authz-v1",
    )
    with pytest.raises(
        RetrievalAuthorizationError,
        match="TENANT_MISMATCH",
    ):
        authorize_document_metadata(source, production)


@pytest.mark.parametrize(
    "record",
    [
        {
            key: value
            for key, value in _chunk_record().items()
            if key != "acl_version"
        },
        _chunk_record(acl_principals=["alice"]),
        _chunk_record(visibility="restricted", acl_principals=[]),
    ],
)
def test_keyword_and_hybrid_reject_partial_or_invalid_acl(
    tmp_path: Path,
    record: dict[str, Any],
) -> None:
    """两种入库路径共享 fail-closed 校验，local 开关也不能修复部分 ACL。"""

    path = _write_jsonl(tmp_path / "invalid.jsonl", record)
    for reader in (
        lambda: list(
            iter_index_actions(
                path,
                "chunks",
                allow_legacy_local=True,
            )
        ),
        lambda: list(
            iter_chunk_batches(
                path,
                8,
                allow_legacy_local=True,
            )
        ),
    ):
        with pytest.raises(RetrievalAuthorizationError):
            reader()


def test_existing_index_requires_acl_mapping_or_explicit_local_upgrade() -> None:
    """生产拒绝旧 strict mapping；显式本地模式只补缺失 ACL keyword 字段。"""

    class FakeIndices:
        """模拟缺少 ACL 字段的既有严格索引。"""

        def __init__(self) -> None:
            """初始化映射更新调用记录。"""

            self.put_mapping_calls: list[dict[str, Any]] = []

        def exists(self, *, index: str) -> bool:
            """报告测试索引已经存在。"""

            return True

        def get_mapping(self, *, index: str) -> dict[str, Any]:
            """返回缺少 ACL 字段的严格映射。"""

            return {
                index: {
                    "mappings": {
                        "dynamic": "strict",
                        "properties": {
                            "chunk_id": {"type": "keyword"}
                        },
                    }
                }
            }

        def put_mapping(
            self,
            *,
            index: str,
            body: dict[str, Any],
        ) -> None:
            """记录本地兼容模式提交的映射补丁。"""

            self.put_mapping_calls.append(body)

    class FakeOpenSearch:
        """暴露映射校验所需的最小索引客户端。"""

        indices = FakeIndices()

    client = FakeOpenSearch()
    with pytest.raises(RuntimeError, match="ACL_INDEX_MAPPING_INCOMPLETE"):
        ensure_chunk_index(
            client,  # type: ignore[arg-type]
            "chunks",
        )

    assert ensure_chunk_index(
        client,  # type: ignore[arg-type]
        "chunks",
        allow_legacy_local=True,
    ) is False
    upgraded = client.indices.put_mapping_calls[0]["properties"]
    assert set(upgraded) == {
        "tenant_id",
        "visibility",
        "acl_principals",
        "acl_version",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "enterprise_agent.cli.data_pipeline",
        "enterprise_agent.cli.multi_source_index",
        "enterprise_agent.cli.keyword_ingest",
        "enterprise_agent.cli.hybrid_ingest",
    ],
)
def test_ingestion_cli_exposes_mutually_exclusive_acl_modes(
    module_name: str,
) -> None:
    """每个摄取 CLI 都公开同名安全参数，避免入口间行为漂移。"""

    completed = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0
    assert b"--expected-tenant-id" in output
    assert b"--allow-legacy-local" in output


def test_cli_rejects_selecting_production_and_local_modes_together() -> None:
    """可信租户和旧本地兼容不能同时选择。"""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "enterprise_agent.cli.data_pipeline",
            "--data-root",
            ".",
            "--expected-tenant-id",
            "tenant-a",
            "--allow-legacy-local",
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
