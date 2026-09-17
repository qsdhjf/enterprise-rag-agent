"""把现有混合检索工具适配为 Agent 的搜索与证据解析接口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from typing import cast

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    SourceName,
)
from enterprise_agent.agent.state import (
    EvidenceReference,
)
from enterprise_agent.agent.tool_executor import (
    SearchToolRequest,
    SearchToolResponse,
)
from enterprise_agent.retrieval.enterprise_search import (
    EnterpriseSearchInput,
    EnterpriseSearchTool,
    SUPPORTED_SOURCES,
)
from enterprise_agent.retrieval.access_control import (
    RetrievalAccessContext,
    authorize_document_metadata,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
    open_search_hit_from_document,
)


def _content_hash(text: str) -> str:
    """计算证据正文的稳定 SHA-256，用于发现缓存内容被替换。"""

    return sha256(
        text.encode("utf-8")
    ).hexdigest()


class LocalEnterpriseSearchAdapter:
    """
    把现有EnterpriseSearchTool适配成Agent搜索接口。

    当前版本用于本地单进程运行：
    - AgentState只保存证据引用；
    - 正文临时保存在内存；
    - 后续生产版再替换成持久Evidence Store。
    """

    def __init__(
        self,
        search_tool: EnterpriseSearchTool,
    ) -> None:
        """绑定可复用检索工具，并初始化本次请求的内存证据缓存。"""

        self._search_tool = search_tool

        self._hits_by_key: dict[
            tuple[str, str, str, str],
            OpenSearchHit,
        ] = {}

    @staticmethod
    def _trusted_principal(
        request: SearchToolRequest,
    ) -> PrincipalContext:
        """取出执行节点注入的主体，并复验旧审计字段没有发生分叉。"""

        principal = request.principal
        if principal is None:
            raise RuntimeError("TRUSTED_SEARCH_PRINCIPAL_REQUIRED")
        if (
            request.tenant_id != principal.tenant_id
            or request.subject_id != principal.subject_id
            or request.authz_version != principal.authz_version
        ):
            raise RuntimeError("SEARCH_PRINCIPAL_MISMATCH")
        return principal

    @staticmethod
    def _authorize_hit(
        hit: OpenSearchHit,
        access_context: RetrievalAccessContext,
    ) -> OpenSearchHit:
        """每次使用缓存正文前重新复验其租户与主体 ACL。"""

        document_metadata: dict[str, object] = {}
        has_explicit_metadata = any(
            value is not None
            for value in (
                hit.tenant_id,
                hit.visibility,
                hit.acl_version,
            )
        )
        if has_explicit_metadata:
            document_metadata = {
                "tenant_id": hit.tenant_id,
                "visibility": hit.visibility,
                "acl_principals": hit.acl_principals,
                "acl_version": hit.acl_version,
            }

        metadata = authorize_document_metadata(
            document_metadata,
            access_context,
        )
        return replace(
            hit,
            tenant_id=metadata.tenant_id,
            visibility=metadata.visibility,
            acl_principals=metadata.acl_principals,
            acl_version=metadata.acl_version,
        )

    def run(
        self,
        request: SearchToolRequest,
    ) -> SearchToolResponse:
        """执行混合检索，把正文缓存起来并仅向图状态返回引用。"""

        principal = self._trusted_principal(request)
        access_context = RetrievalAccessContext.from_principal(
            principal
        )
        result = self._search_tool.run_for_principal(
            EnterpriseSearchInput(
                query=request.query,
                source=request.source,
                top_k=request.top_k,
            ),
            principal=principal,
        )

        references: list[
            EvidenceReference
        ] = []

        for hybrid_hit in result.hits:
            # OpenSearch 查询已带过滤器，但仍逐条复验，防止错误代理或伪后端绕过。
            hit = self._authorize_hit(
                hybrid_hit.hit,
                access_context,
            )

            if hit.source not in SUPPORTED_SOURCES:
                raise RuntimeError(
                    "UNSUPPORTED_HIT_SOURCE"
                )

            if (
                request.source is not None
                and hit.source != request.source
            ):
                raise RuntimeError(
                    "SEARCH_SOURCE_MISMATCH"
                )

            source = cast(
                SourceName,
                hit.source,
            )

            content_hash = _content_hash(
                hit.text
            )

            # 身份与授权版本参与缓存键，并且读取时仍会再次执行 ACL 复验。
            cache_key = (
                request.tenant_id,
                request.subject_id,
                request.authz_version,
                hit.chunk_id,
            )

            self._hits_by_key[
                cache_key
            ] = hit

            # 图状态不保存正文，只保存可校验、可持久化的最小引用信息。
            references.append(
                EvidenceReference(
                    tenant_id=request.tenant_id,
                    source=source,
                    document_id=hit.document_id,
                    chunk_id=hit.chunk_id,
                    sequence=hit.sequence,
                    content_sha256=content_hash,

                    acl_version=cast(str, hit.acl_version),
                )
            )

        return SearchToolResponse(
            evidence=tuple(references),
        )

    def _resolve_from_opensearch(
        self,
        reference: EvidenceReference,
        runtime_context: AgentRuntimeContext,
    ) -> OpenSearchHit:
        """缓存缺失时按 chunk_id 回读索引，并执行失败关闭式复验。"""

        client = getattr(
            self._search_tool,
            "_opensearch_client",
            None,
        )
        index_name = getattr(
            self._search_tool,
            "_index_name",
            None,
        )

        if client is None or not index_name:
            raise RuntimeError(
                "EVIDENCE_NOT_IN_LOCAL_CACHE"
            )

        if (
            reference["source"]
            not in runtime_context.retrieval_policy.allowed_sources
        ):
            raise RuntimeError(
                "EVIDENCE_SOURCE_NOT_ALLOWED"
            )

        response = client.get(
            index=index_name,
            id=reference["chunk_id"],
        )

        if response.get("found") is False:
            raise RuntimeError("EVIDENCE_NOT_FOUND")

        source_document = response.get("_source")

        if not isinstance(source_document, dict):
            raise RuntimeError("EVIDENCE_SOURCE_INVALID")

        return open_search_hit_from_document(
            source_document,
            score=0.0,
            access_context=RetrievalAccessContext.from_principal(
                runtime_context.principal
            ),
        )

    def resolve(
        self,
        references: Sequence[EvidenceReference],
        runtime_context: AgentRuntimeContext,
        *,
        fresh: bool = False,
    ) -> Sequence[OpenSearchHit]:
        """按引用顺序取回正文，并重新验证租户、元数据和内容哈希。"""

        principal = runtime_context.principal
        resolved: list[OpenSearchHit] = []

        for reference in references:
            if (
                reference["tenant_id"]
                != principal.tenant_id
            ):
                raise RuntimeError(
                    "EVIDENCE_TENANT_MISMATCH"
                )

            if (
                reference["source"]
                not in runtime_context.retrieval_policy.allowed_sources
            ):
                raise RuntimeError(
                    "EVIDENCE_SOURCE_NOT_ALLOWED"
                )

            cache_key = (
                reference["tenant_id"],
                principal.subject_id,
                principal.authz_version,
                reference["chunk_id"],
            )

            hit = self._hits_by_key.get(
                cache_key
            )

            if hit is None or fresh:
                hit = self._resolve_from_opensearch(
                    reference,
                    runtime_context,
                )
                self._hits_by_key[cache_key] = hit

            # 即使命中内存缓存，也必须按本轮可信主体重新授权。
            hit = self._authorize_hit(
                hit,
                RetrievalAccessContext.from_principal(principal),
            )

            # 引用与缓存正文必须逐字段匹配，避免错引同名或过期切块。
            if (
                hit.chunk_id
                != reference["chunk_id"]
                or hit.document_id
                != reference["document_id"]
                or hit.source
                != reference["source"]
                or hit.sequence
                != reference["sequence"]
            ):
                raise RuntimeError(
                    "EVIDENCE_REFERENCE_MISMATCH"
                )

            if hit.acl_version != reference["acl_version"]:
                raise RuntimeError(
                    "EVIDENCE_ACL_VERSION_CHANGED"
                )

            # 哈希验证可发现检索后到生成前正文被意外修改的情况。
            actual_hash = _content_hash(
                hit.text
            )

            if (
                actual_hash
                != reference["content_sha256"]
            ):
                raise RuntimeError(
                    "EVIDENCE_CONTENT_CHANGED"
                )

            resolved.append(hit)

        return tuple(resolved)
