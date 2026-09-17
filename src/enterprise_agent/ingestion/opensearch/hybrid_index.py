"""在基础文本映射上增加 HNSW 向量字段，并校验已有混合索引的数据契约。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from opensearchpy import OpenSearch

from enterprise_agent.ingestion.opensearch.index import (
    CHUNK_INDEX_BODY,
    ensure_existing_acl_mapping,
)


def build_hybrid_index_body(
    embedding_model: str,
    dimensions: int,
) -> dict[str, Any]:
    """构造同时支持 BM25 与余弦向量近邻检索的索引配置。

    Args:
        embedding_model: 生成索引向量的模型名称，会写入映射元数据。
        dimensions: 每个向量必须具有的维度。
    """

    if not embedding_model.strip():
        raise ValueError("Embedding模型名称不能为空")

    if dimensions <= 0:
        raise ValueError("向量维度必须大于0")

    # 复制原BM25索引结构，避免修改原来的配置对象。
    body = deepcopy(CHUNK_INDEX_BODY)

    # 启用OpenSearch的向量近邻检索功能。
    body["settings"]["index.knn"] = True

    # 新增由参数决定维度的向量字段；HNSW 用近似搜索换取更低查询延迟。
    body["mappings"]["properties"]["embedding"] = {
        "type": "knn_vector",
        "dimension": dimensions,
        "method": {
            "name": "hnsw",
            "engine": "lucene",
            "space_type": "cosinesimil",
            "parameters": {
                "ef_construction": 100,
                "m": 16,
            },
        },
    }

    # 在索引映射中记录向量的数据契约。
    body["mappings"]["_meta"] = {
        "schema_version": "hybrid-v1",
        "embedding_model": embedding_model,
        "embedding_dimensions": dimensions,
    }

    return body


def ensure_hybrid_index(
    client: OpenSearch,
    index_name: str,
    embedding_model: str,
    dimensions: int,
    *,
    allow_legacy_local: bool = False,
) -> bool:
    """创建混合索引，或验证已有索引与当前向量模型配置兼容。

    已有索引不能在原地安全切换向量维度或模型，因此不一致时明确报错，
    避免把不同向量空间的数据混在同一索引中。

    Returns:
        新建索引时为 ``True``，验证并复用已有索引时为 ``False``。
    """

    if client.indices.exists(index=index_name):
        mapping_response = client.indices.get_mapping(
            index=index_name,
        )
        mapping = mapping_response[index_name]["mappings"]
        ensure_existing_acl_mapping(
            client,
            index_name,
            mapping,
            allow_legacy_local=allow_legacy_local,
        )

        # 先检查字段，再检查维度和模型元数据，错误信息会指向具体契约差异。
        embedding_mapping = mapping.get(
            "properties",
            {},
        ).get("embedding")

        if embedding_mapping is None:
            raise RuntimeError(
                f"已有索引 {index_name} 没有embedding字段"
            )

        actual_dimensions = embedding_mapping.get(
            "dimension"
        )

        if actual_dimensions != dimensions:
            raise RuntimeError(
                "已有索引的向量维度不一致："
                f"期望 {dimensions}，"
                f"实际 {actual_dimensions}"
            )

        index_metadata = mapping.get("_meta", {})
        actual_model = index_metadata.get(
            "embedding_model"
        )

        if actual_model != embedding_model:
            raise RuntimeError(
                "已有索引的Embedding模型不一致："
                f"期望 {embedding_model}，"
                f"实际 {actual_model}"
            )

        return False

    index_body = build_hybrid_index_body(
        embedding_model=embedding_model,
        dimensions=dimensions,
    )

    client.indices.create(
        index=index_name,
        body=index_body,
    )

    return True
