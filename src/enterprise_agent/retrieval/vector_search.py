"""把查询向量化后执行 OpenSearch k-NN 语义检索。"""

from __future__ import annotations

from opensearchpy import OpenSearch

from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.retrieval.access_control import (
    RetrievalAccessContext,
    build_opensearch_search_filter,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
    open_search_hit_from_document,
)


def vector_search_chunks(
    client: OpenSearch,
    index_name: str,
    embedding_client: OpenAICompatibleEmbeddingClient,
    query: str,
    top_k: int = 5,
    source: str | None = None,
    max_chunks_per_document: int = 2,
    access_context: RetrievalAccessContext | None = None,
) -> list[OpenSearchHit]:
    """按向量相似度搜索文本块，并控制每篇文档最多返回的块数。

    查询可按数据源过滤；候选数会扩大到目标数的四倍（最多 100），
    以便文档级去重后仍尽量凑足 ``top_k``。
    """

    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if max_chunks_per_document <= 0:
        raise ValueError(
            "max_chunks_per_document 必须大于 0"
        )

    # 维度由客户端配置校验，不在此处硬编码具体模型维度。
    query_vector = embedding_client.embed_text(query)

    # 多召回一些候选切块，方便后面按文档去重。
    candidate_k = min(top_k * 4, 100)
    # 旧签名固定映射到本地演示租户，生产调用必须显式传入可信授权上下文。
    effective_access = (
        access_context or RetrievalAccessContext.local_demo()
    )

    # 这些参数直接对应 OpenSearch knn 查询中 embedding 字段的配置。
    knn_parameters: dict = {
        "vector": query_vector,
        "k": candidate_k,
        "filter": build_opensearch_search_filter(
            effective_access,
            source,
        ),
    }

    request_body = {
        "size": candidate_k,

        # 返回结果时排除高维向量，减少网络传输。
        "_source": {
            "excludes": ["embedding"],
        },

        "query": {
            "knn": {
                "embedding": knn_parameters,
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

        current_count = document_counts.get(
            document_id,
            0,
        )

        # 避免同一篇长文档的相邻块占满所有搜索结果。
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
