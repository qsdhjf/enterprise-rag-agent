"""把文本块 JSONL 转为 OpenSearch bulk 动作并进行流式批量写入。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import streaming_bulk
from enterprise_agent.ingestion.identity import normalize_chunk_identity

from enterprise_agent.ingestion.access_metadata import (
    normalize_index_document_acl,
)
from enterprise_agent.retrieval.access_control import (
    RetrievalAuthorizationError,
)


@dataclass(frozen=True)
class BulkIndexResult:
    """一次批量写入中成功和失败的文档数量。"""

    succeeded: int
    failed: int


def iter_index_actions(
    jsonl_path: Path,
    index_name: str,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> Iterator[dict[str, Any]]:
    """逐行校验 JSONL，并产出 ``streaming_bulk`` 所需的动作字典。

    空行会被忽略；非法 JSON 或缺少 ``chunk_id`` 的行会带行号抛错，便于定位。
    ``chunk_id`` 同时作为 OpenSearch 文档 ID，使重复摄取成为覆盖而非新增副本。
    """

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 第 {line_number} 行不是合法 JSON"
                ) from error

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"JSONL 第 {line_number} 行必须是对象"
                )

            chunk_id = chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"JSONL 第 {line_number} 行缺少 chunk_id"
                )

            try:
                chunk = normalize_index_document_acl(
                    chunk,
                    expected_tenant_id=expected_tenant_id,
                    allow_legacy_local=allow_legacy_local,
                )
            except RetrievalAuthorizationError as error:
                raise RetrievalAuthorizationError(
                    f"{error}:JSONL_LINE_{line_number}"
                ) from error

            chunk = normalize_chunk_identity(chunk)
            chunk_id = chunk["chunk_id"]
            # 摄取时补充版本与 UTC 时间，源 JSONL 无需承担运行时元数据。
            chunk["schema_version"] = "v1"
            chunk["ingested_at"] = datetime.now(timezone.utc).isoformat()

            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": chunk_id,
                "_source": chunk,
            }


def bulk_index_jsonl(
    client: OpenSearch,
    index_name: str,
    jsonl_path: Path,
    batch_size: int = 100,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> BulkIndexResult:
    """以有限内存流式写入 JSONL，并刷新索引后返回成功/失败计数。

    单条失败不会立即终止整个批次；底层 helper 会按配置重试并把结果逐条返回。
    """

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    succeeded = 0
    failed = 0

    actions = iter_index_actions(
        jsonl_path=jsonl_path,
        index_name=index_name,
        expected_tenant_id=expected_tenant_id,
        allow_legacy_local=allow_legacy_local,
    )

    # streaming_bulk 消费上面的生成器，无需把整个文件转换成动作列表。
    for ok, _ in streaming_bulk(
        client=client,
        actions=actions,
        chunk_size=batch_size,
        max_retries=3,
        initial_backoff=1,
        max_backoff=8,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            succeeded += 1
        else:
            failed += 1

    # 显式刷新让函数返回后新文档立即可被搜索，适合离线构建/验证流程。
    client.indices.refresh(index=index_name)

    return BulkIndexResult(
        succeeded=succeeded,
        failed=failed,
    )
