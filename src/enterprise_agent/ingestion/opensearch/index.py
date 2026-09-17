"""定义基础文本块索引映射，并以幂等方式确保索引存在。"""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch


# 映射使用 strict，意外字段会在写入时尽早暴露，而不是悄悄改变索引结构。
CHUNK_INDEX_BODY: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s",
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {
                "type": "keyword",
            },
            "document_id": {
                "type": "keyword",
            },
            "source": {
                "type": "keyword",
            },
            "title": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512,
                    }
                },
            },
            "sequence": {
                "type": "integer",
            },
            "text": {
                "type": "text",
            },
            "created_at": {
                "type": "keyword",
                "ignore_above": 128,
            },
            "updated_at": {
                "type": "keyword",
                "ignore_above": 128,
            },
            "created_at_normalized": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
            "updated_at_normalized": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
            "original_location": {
                "type": "keyword",
                "ignore_above": 2048,
            },
            "tenant_id": {
                "type": "keyword",
            },
            "visibility": {
                "type": "keyword",
            },
            "acl_principals": {
                "type": "keyword",
            },
            "acl_version": {
                "type": "keyword",
            },
            "schema_version": {
                "type": "keyword",
            },
            "ingested_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
        },
    },
}


ACL_PROPERTY_MAPPINGS: dict[str, dict[str, str]] = {
    "tenant_id": {"type": "keyword"},
    "visibility": {"type": "keyword"},
    "acl_principals": {"type": "keyword"},
    "acl_version": {"type": "keyword"},
}


def ensure_existing_acl_mapping(
    client: OpenSearch,
    index_name: str,
    mapping: dict[str, Any],
    *,
    allow_legacy_local: bool = False,
) -> None:
    """验证已有索引的 ACL 字段；仅显式本地模式可补齐缺失 keyword 映射。"""

    properties = mapping.get("properties", {})
    missing = [
        field_name
        for field_name in ACL_PROPERTY_MAPPINGS
        if field_name not in properties
    ]
    invalid = [
        field_name
        for field_name in ACL_PROPERTY_MAPPINGS
        if field_name in properties
        and (
            not isinstance(properties[field_name], dict)
            or properties[field_name].get("type") != "keyword"
        )
    ]
    if invalid:
        raise RuntimeError(
            "ACL_INDEX_MAPPING_INVALID:"
            f"{','.join(sorted(invalid))}; 请重建索引"
        )
    if not missing:
        return
    if not allow_legacy_local:
        raise RuntimeError(
            "ACL_INDEX_MAPPING_INCOMPLETE:"
            f"{','.join(sorted(missing))}; 生产索引必须重建"
        )

    # OpenSearch 允许为现有 strict mapping 新增字段。本地显式兼容时只补固定
    # ACL keyword 字段，不改变其他映射；旧文档仍由 local-demo 查询分支复验。
    client.indices.put_mapping(
        index=index_name,
        body={
            "properties": {
                field_name: ACL_PROPERTY_MAPPINGS[field_name]
                for field_name in missing
            }
        },
    )


def ensure_chunk_index(
    client: OpenSearch,
    index_name: str,
    *,
    allow_legacy_local: bool = False,
) -> bool:
    """在索引不存在时创建基础文本块索引。

    Returns:
        本次确实创建了索引时为 ``True``；已存在时为 ``False``。
    """

    # 先检查使函数可重复执行，避免每次启动都触发“索引已存在”错误。
    if client.indices.exists(index=index_name):
        mapping_response = client.indices.get_mapping(index=index_name)
        ensure_existing_acl_mapping(
            client,
            index_name,
            mapping_response[index_name]["mappings"],
            allow_legacy_local=allow_legacy_local,
        )
        return False

    client.indices.create(
        index=index_name,
        body=CHUNK_INDEX_BODY,
    )

    return True
