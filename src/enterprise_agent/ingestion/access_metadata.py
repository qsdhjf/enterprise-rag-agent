"""在文档读取和索引写入之间统一解析、校验并规范化 ACL 元数据。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from enterprise_agent.retrieval.access_control import (
    LOCAL_DEMO_ACL_VERSION,
    LOCAL_DEMO_TENANT_ID,
    AuthorizedDocumentMetadata,
    RetrievalAuthorizationError,
    normalize_document_acl_metadata,
)


ACL_FIELD_NAMES = frozenset(
    {
        "tenant_id",
        "visibility",
        "acl_principals",
        "acl_version",
    }
)


def local_demo_acl_metadata() -> AuthorizedDocumentMetadata:
    """返回唯一允许用于旧本地样本的完整 ACL 组合。"""

    return AuthorizedDocumentMetadata(
        tenant_id=LOCAL_DEMO_TENANT_ID,
        visibility="local-demo",
        acl_principals=(),
        acl_version=LOCAL_DEMO_ACL_VERSION,
    )


def _candidate_metadata(
    container: Mapping[str, object],
    *,
    location: str,
    expected_tenant_id: str | None,
    explicit_container: bool = False,
    ignore_lone_top_level_visibility: bool = False,
) -> AuthorizedDocumentMetadata | None:
    """从一个允许的位置读取完整 ACL；部分声明一律失败关闭。"""

    present = ACL_FIELD_NAMES.intersection(container)
    if (
        ignore_lone_top_level_visibility
        and present == {"visibility"}
    ):
        # 旧生成数据中的 visibility 是业务描述而非访问控制。只有调用显式
        # local-demo helper 时才允许忽略这个历史字段。
        return None
    if not present:
        if explicit_container:
            raise RetrievalAuthorizationError(
                f"EVIDENCE_ACL_METADATA_MISSING:{location}"
            )
        return None
    if present != ACL_FIELD_NAMES:
        missing = ",".join(sorted(ACL_FIELD_NAMES - present))
        raise RetrievalAuthorizationError(
            "EVIDENCE_ACL_METADATA_INCOMPLETE:"
            f"{location}:missing={missing}"
        )
    return normalize_document_acl_metadata(
        container,
        expected_tenant_id=expected_tenant_id,
    )


def extract_document_acl_metadata(
    raw: Mapping[str, object],
    *,
    expected_tenant_id: str | None = None,
    allow_local_demo_defaults: bool = False,
) -> AuthorizedDocumentMetadata:
    """从源 JSON 的顶层或明确 metadata 区域提取无冲突 ACL。

    支持顶层、access_control、metadata 的直接 ACL 字段，以及
    metadata.access_control。多个位置同时声明时必须规范化后完全一致。
    allow_local_demo_defaults 只供名称明确的本地开发入口使用。
    """

    if allow_local_demo_defaults:
        if (
            expected_tenant_id is not None
            and expected_tenant_id != LOCAL_DEMO_TENANT_ID
        ):
            raise ValueError(
                "allow_local_demo_defaults 只能用于 local-demo-tenant"
            )
        effective_tenant_id = LOCAL_DEMO_TENANT_ID
    else:
        effective_tenant_id = expected_tenant_id

    candidates: list[tuple[str, AuthorizedDocumentMetadata]] = []

    metadata_value = raw.get("metadata")
    has_explicit_nested_acl = (
        "access_control" in raw
        or (
            isinstance(metadata_value, Mapping)
            and (
                "access_control" in metadata_value
                or ACL_FIELD_NAMES.issubset(metadata_value)
            )
        )
    )
    top_level = _candidate_metadata(
        raw,
        location="top-level",
        expected_tenant_id=effective_tenant_id,
        ignore_lone_top_level_visibility=(
            allow_local_demo_defaults or has_explicit_nested_acl
        ),
    )
    if top_level is not None:
        candidates.append(("top-level", top_level))

    access_control_value = raw.get("access_control")
    if "access_control" in raw:
        if not isinstance(access_control_value, Mapping):
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_METADATA_INVALID:access_control"
            )
        nested = _candidate_metadata(
            access_control_value,
            location="access_control",
            expected_tenant_id=effective_tenant_id,
            explicit_container=True,
        )
        if nested is not None:
            candidates.append(("access_control", nested))

    if isinstance(metadata_value, Mapping):
        direct_metadata = _candidate_metadata(
            metadata_value,
            location="metadata",
            expected_tenant_id=effective_tenant_id,
        )
        if direct_metadata is not None:
            candidates.append(("metadata", direct_metadata))

        nested_value = metadata_value.get("access_control")
        if "access_control" in metadata_value:
            if not isinstance(nested_value, Mapping):
                raise RetrievalAuthorizationError(
                    "EVIDENCE_ACL_METADATA_INVALID:"
                    "metadata.access_control"
                )
            nested_metadata = _candidate_metadata(
                nested_value,
                location="metadata.access_control",
                expected_tenant_id=effective_tenant_id,
                explicit_container=True,
            )
            if nested_metadata is not None:
                candidates.append(
                    ("metadata.access_control", nested_metadata)
                )

    if not candidates:
        if allow_local_demo_defaults:
            return local_demo_acl_metadata()
        raise RetrievalAuthorizationError(
            "EVIDENCE_ACL_METADATA_MISSING"
        )

    first_location, first = candidates[0]
    for location, candidate in candidates[1:]:
        if candidate != first:
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_METADATA_CONFLICT:"
                f"{first_location}!={location}"
            )
    return first


def normalize_index_document_acl(
    document: Mapping[str, Any],
    *,
    expected_tenant_id: str | None = None,
    allow_legacy_local: bool = False,
) -> dict[str, Any]:
    """校验 JSONL 记录并返回可安全写入 OpenSearch 的副本。

    兼容模式只接受四个字段全部缺失的旧记录，并补成完整 local-demo 元数据；
    部分字段、空值或生产租户字段不会被静默覆盖。
    """

    present = ACL_FIELD_NAMES.intersection(document)
    if allow_legacy_local:
        if (
            expected_tenant_id is not None
            and expected_tenant_id != LOCAL_DEMO_TENANT_ID
        ):
            raise ValueError(
                "allow_legacy_local 不能与其他 expected_tenant_id 同时使用"
            )
        effective_tenant_id = LOCAL_DEMO_TENANT_ID
    else:
        effective_tenant_id = expected_tenant_id

    if not present:
        if not allow_legacy_local:
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_METADATA_MISSING"
            )
        metadata = local_demo_acl_metadata()
    else:
        if present != ACL_FIELD_NAMES:
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_METADATA_INCOMPLETE"
            )
        metadata = normalize_document_acl_metadata(
            document,
            expected_tenant_id=effective_tenant_id,
        )

    normalized = dict(document)
    normalized.update(
        {
            "tenant_id": metadata.tenant_id,
            "visibility": metadata.visibility,
            "acl_principals": list(metadata.acl_principals),
            "acl_version": metadata.acl_version,
        }
    )
    return normalized
