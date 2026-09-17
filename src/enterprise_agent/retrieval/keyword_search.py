"""使用 OpenSearch BM25 搜索文本块，并限制单篇文档占用的结果数。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from opensearchpy import OpenSearch

from enterprise_agent.retrieval.access_control import (
    RetrievalAccessContext,
    authorize_document_metadata,
    build_opensearch_search_filter,
)


@dataclass(frozen=True)
class OpenSearchHit:
    """从 OpenSearch 命中记录中提取出的稳定业务字段。"""

    score: float
    chunk_id: str
    document_id: str
    source: str
    title: str
    sequence: int
    text: str
    created_at: str | None
    original_location: str | None
    tenant_id: str | None = None
    visibility: str | None = None
    acl_principals: tuple[str, ...] = ()
    acl_version: str | None = None


def open_search_hit_from_document(
    source_document: Mapping[str, object],
    *,
    score: float,
    access_context: RetrievalAccessContext,
) -> OpenSearchHit:
    """把一条 OpenSearch 文档转为已完成租户/ACL 复验的稳定命中。"""

    metadata = authorize_document_metadata(
        source_document,
        access_context,
    )
    required_text_fields = (
        "chunk_id",
        "document_id",
        "source",
        "title",
        "text",
    )
    for field in required_text_fields:
        value = source_document.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError("EVIDENCE_SOURCE_INVALID")

    sequence = source_document.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise RuntimeError("EVIDENCE_SOURCE_INVALID")

    document: dict[str, Any] = dict(source_document)
    return OpenSearchHit(
        score=float(score),
        chunk_id=document["chunk_id"],
        document_id=document["document_id"],
        source=document["source"],
        title=document["title"],
        sequence=sequence,
        text=document["text"],
        created_at=(
            str(document["created_at"])
            if document.get("created_at") is not None
            else None
        ),
        original_location=(
            str(document["original_location"])
            if document.get("original_location") is not None
            else None
        ),
        tenant_id=metadata.tenant_id,
        visibility=metadata.visibility,
        acl_principals=metadata.acl_principals,
        acl_version=metadata.acl_version,
    )


def search_chunks(
    client: OpenSearch,
    index_name: str,
    query: str,
    top_k: int = 5,
    source: str | None = None,
    max_chunks_per_document: int = 2,
    access_context: RetrievalAccessContext | None = None,
) -> list[OpenSearchHit]:
    """按关键词相关性搜索文本块，并返回最多 ``top_k`` 条结果。

    标题权重是正文的三倍；若指定 ``source``，会添加精确来源过滤。
    为提高结果多样性，函数先多召回候选，再限制每篇文档保留的块数。
    """

    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if max_chunks_per_document <= 0:
        raise ValueError("max_chunks_per_document 必须大于 0")

    # 旧签名没有身份参数；它只映射到固定本地演示租户，不能读取生产租户。
    effective_access = (
        access_context or RetrievalAccessContext.local_demo()
    )
    search_filter = build_opensearch_search_filter(
        effective_access,
        source,
    )

    request_body = {
        # 多取候选，给后续“每篇文档最多 N 块”的去重留出空间。
        "size": min(top_k * 4, 100),
        # 向量字段体积大且这里不会使用，排除后可减少响应传输量。
        "_source": {
            "excludes": ["embedding"],
        },
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                # 标题通常更能概括主题，因此赋予三倍权重。
                                "title^3",
                                "text",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": search_filter,
            }
        },
    }

    response = client.search(
        index=index_name,
        body=request_body,
    )

    results: list[OpenSearchHit] = []
    document_counts: dict[str, int] = {}

    for hit in response["hits"]["hits"]:
        source_document = hit["_source"]
        document_id = source_document["document_id"]

        current_count = document_counts.get(document_id, 0)

        # 防止同一篇长文档的相邻块挤占全部结果位置。
        if current_count >= max_chunks_per_document:
            continue

        document_counts[document_id] = current_count + 1

        results.append(
            open_search_hit_from_document(
                source_document,
                score=float(hit["_score"]),
                access_context=effective_access,
            )
        )

        if len(results) >= top_k:
            break

    return results
