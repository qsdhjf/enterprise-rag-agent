"""分批生成文本向量，并将正文、元数据和向量一起写入混合索引。"""

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
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.retrieval.access_control import (
    RetrievalAuthorizationError,
)


@dataclass(frozen=True, slots=True)
class HybridIndexResult:
    """混合摄取过程中已向量化、写入成功和写入失败的数量。"""

    embedded: int
    succeeded: int
    failed: int


def iter_chunk_batches(
    jsonl_path: Path,
    batch_size: int,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> Iterator[list[dict[str, Any]]]:
    """逐行读取并校验 JSONL，按指定大小产出块列表。

    最后一批即使不足 ``batch_size`` 也会在文件结束后产出。
    """

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于0")

    batch: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL第 {line_number} 行不是合法JSON"
                ) from error

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"JSONL第 {line_number} 行必须是对象"
                )

            chunk_id = chunk.get("chunk_id")
            text = str(chunk.get("text") or "").strip()

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"JSONL第 {line_number} 行缺少chunk_id"
                )

            if not text:
                raise ValueError(
                    f"JSONL第 {line_number} 行正文为空"
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

            batch.append(normalize_chunk_identity(chunk))

            if len(batch) >= batch_size:
                yield batch
                batch = []

    # 文件末尾仍有记录时不要丢弃不足一整批的尾批次。
    if batch:
        yield batch


def build_embedding_text(
    chunk: dict[str, Any],
) -> str:
    """组合标题和正文作为向量模型输入；无标题时只返回正文。"""

    title = str(chunk.get("title") or "").strip()
    text = str(chunk.get("text") or "").strip()

    if title:
        return f"title: {title}\n\ntext: {text}"

    return text


def bulk_index_hybrid_jsonl(
    client: OpenSearch,
    index_name: str,
    jsonl_path: Path,
    embedding_client: OpenAICompatibleEmbeddingClient,
    embedding_batch_size: int = 8,
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> HybridIndexResult:
    """批量向量化 JSONL 文本块，并流式写入 OpenSearch 混合索引。

    数据先按 ``embedding_batch_size`` 分批请求模型，再把每批向量与原块严格
    对齐成 bulk 动作。``zip(..., strict=True)`` 会在响应数量不一致时立即失败，
    防止向量被错配给其他文档。
    """

    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"JSONL文件不存在：{jsonl_path}"
        )

    embedded = 0
    succeeded = 0
    failed = 0

    batches = iter_chunk_batches(
        jsonl_path=jsonl_path,
        batch_size=embedding_batch_size,
        expected_tenant_id=expected_tenant_id,
        allow_legacy_local=allow_legacy_local,
    )

    for chunks in batches:
        # 标题常包含高价值主题词，将其与正文共同编码可改善语义召回。
        embedding_texts = [
            build_embedding_text(chunk)
            for chunk in chunks
        ]

        vectors = embedding_client.embed_texts(
            embedding_texts
        )

        embedded += len(vectors)
        ingested_at = datetime.now(
            timezone.utc
        ).isoformat()

        actions: list[dict[str, Any]] = []

        # strict=True 是关键的数据完整性保护：块数必须与向量数完全一致。
        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
            # 复制后再补字段，避免修改当前批次中保存的原始解析结果。
            document = dict(chunk)
            document["embedding"] = vector
            document["schema_version"] = "hybrid-v1"
            document["ingested_at"] = ingested_at

            actions.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": document["chunk_id"],
                    "_source": document,
                }
            )

        # 每个向量批次独立提交，既控制内存，也使失败计数可逐条累积。
        for ok, _ in streaming_bulk(
            client=client,
            actions=actions,
            chunk_size=len(actions),
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

    # 确保调用返回后，本次写入的向量文档立即可检索。
    client.indices.refresh(index=index_name)

    return HybridIndexResult(
        embedded=embedded,
        succeeded=succeeded,
        failed=failed,
    )
