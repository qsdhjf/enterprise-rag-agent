"""构造统一的 OpenSearch 租户/主体 ACL 过滤器并复验返回文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


LOCAL_DEMO_TENANT_ID = "local-demo-tenant"
LOCAL_DEMO_SUBJECT_ID = "local-demo-user"
LOCAL_DEMO_ACL_VERSION = "local-dev-v1"

# 只有这些规范化值代表“当前租户内所有已认证主体可读”。其他可见性值都必须
# 通过 acl_principals 精确命中，避免把自由文本 visibility 当成授权规则。
TENANT_WIDE_VISIBILITIES = frozenset(
    {"tenant", "internal", "public"}
)
RESTRICTED_VISIBILITIES = frozenset({"restricted", "private"})
SUPPORTED_VISIBILITIES = (
    TENANT_WIDE_VISIBILITIES
    | RESTRICTED_VISIBILITIES
    | {"local-demo"}
)


class PrincipalLike(Protocol):
    """检索层需要的最小可信主体合同。"""

    tenant_id: str
    subject_id: str
    roles: frozenset[str]
    groups: frozenset[str]
    authz_version: str


class RetrievalAuthorizationError(RuntimeError):
    """表示查询过滤或返回文档未通过租户/主体授权检查。"""


@dataclass(frozen=True, slots=True)
class RetrievalAccessContext:
    """从服务端可信主体派生的检索授权快照。"""

    tenant_id: str
    subject_id: str
    roles: frozenset[str]
    groups: frozenset[str]
    authz_version: str

    @classmethod
    def from_principal(
        cls,
        principal: PrincipalLike,
    ) -> RetrievalAccessContext:
        """复制可信 PrincipalContext，禁止从查询正文解析授权字段。"""

        return cls(
            tenant_id=principal.tenant_id,
            subject_id=principal.subject_id,
            roles=frozenset(principal.roles),
            groups=frozenset(principal.groups),
            authz_version=principal.authz_version,
        )

    @classmethod
    def local_demo(cls) -> RetrievalAccessContext:
        """为保留旧 CLI 行为构造唯一允许读取无 ACL 旧索引的身份。"""

        return cls(
            tenant_id=LOCAL_DEMO_TENANT_ID,
            subject_id=LOCAL_DEMO_SUBJECT_ID,
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="local-demo-authz-v1",
        )

    @property
    def principal_tokens(self) -> frozenset[str]:
        """生成不会混淆用户、角色和用户组命名空间的 ACL token。"""

        return frozenset(
            {
                f"user:{self.subject_id}",
                *(f"role:{role}" for role in self.roles),
                *(f"group:{group}" for group in self.groups),
            }
        )

    @property
    def allows_legacy_local_document(self) -> bool:
        """只有固定本地演示租户可读取完全缺失授权元数据的旧文档。"""

        return self.tenant_id == LOCAL_DEMO_TENANT_ID


@dataclass(frozen=True, slots=True)
class AuthorizedDocumentMetadata:
    """文档通过授权后得到的规范化租户与 ACL 元数据。"""

    tenant_id: str
    visibility: str
    acl_principals: tuple[str, ...]
    acl_version: str


def build_opensearch_access_filter(
    access_context: RetrievalAccessContext,
) -> dict[str, Any]:
    """构造租户与主体 ACL 必须同时满足的 OpenSearch 过滤器。"""

    visibility_or_acl: list[dict[str, Any]] = [
        {
            "terms": {
                "visibility": sorted(TENANT_WIDE_VISIBILITIES)
            }
        },
        {
            "terms": {
                "acl_principals": sorted(
                    access_context.principal_tokens
                )
            }
        },
    ]
    if access_context.allows_legacy_local_document:
        # 新生成的本地开发 JSONL 会显式携带这组标记；它只能在固定 demo
        # 租户下匹配，不能成为生产租户的通用 visibility。
        visibility_or_acl.append(
            {
                "bool": {
                    "filter": [
                        {"term": {"visibility": "local-demo"}},
                        {
                            "term": {
                                "acl_version": LOCAL_DEMO_ACL_VERSION
                            }
                        },
                    ]
                }
            }
        )

    authorized_document: dict[str, Any] = {
        "bool": {
            "filter": [
                {
                    "term": {
                        "tenant_id": access_context.tenant_id,
                    }
                },
                {"exists": {"field": "acl_version"}},
                {
                    "bool": {
                        "should": visibility_or_acl,
                        "minimum_should_match": 1,
                    }
                },
            ]
        }
    }

    if not access_context.allows_legacy_local_document:
        return authorized_document

    # 旧样本索引完全没有这些字段。兼容分支要求四项全部缺失，部分迁移或损坏的
    # 文档不会被误判成旧格式；而且该分支只为固定 local-demo 租户生成。
    legacy_local_document: dict[str, Any] = {
        "bool": {
            "must_not": [
                {"exists": {"field": field}}
                for field in (
                    "tenant_id",
                    "visibility",
                    "acl_principals",
                    "acl_version",
                )
            ]
        }
    }
    return {
        "bool": {
            "should": [
                authorized_document,
                legacy_local_document,
            ],
            "minimum_should_match": 1,
        }
    }


def build_opensearch_search_filter(
    access_context: RetrievalAccessContext,
    source: str | None,
) -> dict[str, Any]:
    """组合统一授权过滤器与可选来源过滤器，供 BM25/k-NN 共用。"""

    clauses: list[dict[str, Any]] = [
        build_opensearch_access_filter(access_context)
    ]
    if source:
        clauses.append({"term": {"source": source}})
    return {"bool": {"filter": clauses}}


def _normalize_acl_principals(value: object) -> tuple[str, ...]:
    """把 keyword 单值或数组规范化为去重 tuple，非法类型失败关闭。"""

    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        candidates = tuple(value)
    else:
        raise RetrievalAuthorizationError(
            "EVIDENCE_ACL_PRINCIPALS_INVALID"
        )

    normalized: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_PRINCIPALS_INVALID"
            )
        token = candidate.strip()
        prefix, separator, identifier = token.partition(":")
        if (
            separator != ":"
            or prefix not in {"user", "role", "group"}
            or not identifier.strip()
        ):
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_PRINCIPALS_INVALID"
            )
        normalized.append(f"{prefix}:{identifier.strip()}")
    return tuple(sorted(set(normalized)))


def normalize_document_acl_metadata(
    document: Mapping[str, object],
    *,
    expected_tenant_id: str | None = None,
) -> AuthorizedDocumentMetadata:
    """严格解析显式 ACL 字段，供摄取与检索返回值复用同一数据合同。"""

    acl_fields = {
        "tenant_id",
        "visibility",
        "acl_principals",
        "acl_version",
    }
    present_fields = {
        field for field in acl_fields if field in document
    }
    if not present_fields:
        raise RetrievalAuthorizationError(
            "EVIDENCE_ACL_METADATA_MISSING"
        )
    if present_fields != acl_fields:
        raise RetrievalAuthorizationError(
            "EVIDENCE_ACL_METADATA_INCOMPLETE"
        )

    tenant_id = document["tenant_id"]
    visibility = document["visibility"]
    acl_version = document["acl_version"]
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise RetrievalAuthorizationError("EVIDENCE_TENANT_INVALID")
    normalized_tenant = tenant_id.strip()
    if (
        expected_tenant_id is not None
        and normalized_tenant != expected_tenant_id
    ):
        raise RetrievalAuthorizationError("EVIDENCE_TENANT_MISMATCH")
    if not isinstance(visibility, str) or not visibility.strip():
        raise RetrievalAuthorizationError("EVIDENCE_VISIBILITY_INVALID")
    normalized_visibility = visibility.strip().lower()
    if normalized_visibility not in SUPPORTED_VISIBILITIES:
        raise RetrievalAuthorizationError("EVIDENCE_VISIBILITY_INVALID")
    if not isinstance(acl_version, str) or not acl_version.strip():
        raise RetrievalAuthorizationError("EVIDENCE_ACL_VERSION_INVALID")
    normalized_acl_version = acl_version.strip()
    acl_principals = _normalize_acl_principals(
        document["acl_principals"]
    )

    if (
        normalized_visibility in RESTRICTED_VISIBILITIES
        and not acl_principals
    ):
        raise RetrievalAuthorizationError(
            "EVIDENCE_ACL_PRINCIPALS_REQUIRED"
        )
    if normalized_visibility == "local-demo" and (
        normalized_tenant != LOCAL_DEMO_TENANT_ID
        or normalized_acl_version != LOCAL_DEMO_ACL_VERSION
        or acl_principals
    ):
        raise RetrievalAuthorizationError(
            "EVIDENCE_LOCAL_DEMO_ACL_INVALID"
        )

    return AuthorizedDocumentMetadata(
        tenant_id=normalized_tenant,
        visibility=normalized_visibility,
        acl_principals=acl_principals,
        acl_version=normalized_acl_version,
    )


def authorize_document_metadata(
    document: Mapping[str, object],
    access_context: RetrievalAccessContext,
) -> AuthorizedDocumentMetadata:
    """复验 OpenSearch 返回文档；过滤器失效或被绕过时仍然失败关闭。"""

    acl_fields = {
        "tenant_id",
        "visibility",
        "acl_principals",
        "acl_version",
    }
    present_fields = {
        field for field in acl_fields if field in document
    }

    if not present_fields:
        if not access_context.allows_legacy_local_document:
            raise RetrievalAuthorizationError(
                "EVIDENCE_ACL_METADATA_MISSING"
            )
        return AuthorizedDocumentMetadata(
            tenant_id=LOCAL_DEMO_TENANT_ID,
            visibility="local-demo",
            acl_principals=(),
            acl_version=LOCAL_DEMO_ACL_VERSION,
        )

    metadata = normalize_document_acl_metadata(
        document,
        expected_tenant_id=access_context.tenant_id,
    )
    synthesized_local_legacy = (
        access_context.allows_legacy_local_document
        and metadata.tenant_id == LOCAL_DEMO_TENANT_ID
        and metadata.visibility == "local-demo"
        and metadata.acl_version == LOCAL_DEMO_ACL_VERSION
        and not metadata.acl_principals
    )
    tenant_wide = (
        metadata.visibility in TENANT_WIDE_VISIBILITIES
    )
    explicitly_allowed = bool(
        set(metadata.acl_principals)
        & access_context.principal_tokens
    )

    if (
        not synthesized_local_legacy
        and not tenant_wide
        and not explicitly_allowed
    ):
        raise RetrievalAuthorizationError("EVIDENCE_ACCESS_DENIED")

    return metadata
