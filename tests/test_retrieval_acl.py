"""验证企业检索从查询过滤到证据解析都执行租户和主体 ACL。"""

from __future__ import annotations

from hashlib import sha256

import pytest

import enterprise_agent.retrieval.enterprise_search as enterprise_search_module
import enterprise_agent.retrieval.hybrid_search as hybrid_search_module
from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.search_adapter import LocalEnterpriseSearchAdapter
from enterprise_agent.agent.state import EvidenceReference
from enterprise_agent.agent.tool_executor import SearchToolRequest
from enterprise_agent.retrieval.access_control import (
    LOCAL_DEMO_ACL_VERSION,
    LOCAL_DEMO_TENANT_ID,
    RetrievalAccessContext,
    RetrievalAuthorizationError,
    build_opensearch_search_filter,
)
from enterprise_agent.retrieval.enterprise_search import (
    EnterpriseSearchInput,
    EnterpriseSearchResult,
    EnterpriseSearchTool,
)
from enterprise_agent.retrieval.hybrid_search import HybridSearchHit
from enterprise_agent.retrieval.hybrid_search import hybrid_search_chunks
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
    search_chunks,
)
from enterprise_agent.retrieval.vector_search import vector_search_chunks


def make_principal(
    *,
    tenant_id: str = "tenant-a",
    subject_id: str = "alice",
    roles: frozenset[str] = frozenset({"employee"}),
    groups: frozenset[str] = frozenset({"engineering"}),
) -> PrincipalContext:
    """构造服务端认证系统提供的可信主体。"""

    return PrincipalContext(
        tenant_id=tenant_id,
        subject_id=subject_id,
        roles=roles,
        groups=groups,
        authz_version="authz-v1",
    )


def make_context(
    principal: PrincipalContext | None = None,
) -> AgentRuntimeContext:
    """构造证据解析阶段使用的可信运行上下文。"""

    return AgentRuntimeContext(
        request_id="acl-test-001",
        principal=principal or make_principal(),
        budget=RunBudget(),
        retrieval_policy=RetrievalPolicy(
            allowed_sources=frozenset({"jira"}),
        ),
    )


def make_document(**changes: object) -> dict[str, object]:
    """构造一条带完整生产授权元数据的 OpenSearch 文档。"""

    document: dict[str, object] = {
        "chunk_id": "chunk-001",
        "document_id": "document-001",
        "source": "jira",
        "title": "Deployment record",
        "sequence": 0,
        "text": "The approved deployment date is Friday.",
        "created_at": None,
        "original_location": None,
        "tenant_id": "tenant-a",
        "visibility": "restricted",
        "acl_principals": ["group:engineering"],
        "acl_version": "acl-v1",
    }
    document.update(changes)
    return document


class FakeOpenSearch:
    """记录 search/get 请求，并允许模拟错误后端返回未过滤文档。"""

    def __init__(self, documents=()) -> None:
        """保存测试文档并初始化请求记录。"""

        self.documents = tuple(documents)
        self.search_bodies: list[dict[str, object]] = []
        self.get_calls: list[tuple[str, str]] = []

    def search(self, *, index: str, body: dict[str, object]):
        """记录检索请求并返回所有预设文档。"""

        self.search_bodies.append(body)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 1.0,
                        "_source": document,
                    }
                    for document in self.documents
                ]
            }
        }

    def get(self, *, index: str, id: str):
        """记录按 ID 读取并返回首个预设文档。"""

        self.get_calls.append((index, id))
        if not self.documents:
            return {"found": False}
        return {"found": True, "_source": self.documents[0]}


class FakeEmbeddingClient:
    """返回固定查询向量，便于检查 k-NN 请求体。"""

    def embed_text(self, text: str) -> list[float]:
        """返回固定二维向量。"""

        return [0.1, 0.2]


def make_hybrid_hit(hit: OpenSearchHit) -> HybridSearchHit:
    """把一条业务命中包装成 EnterpriseSearchResult 所需结构。"""

    return HybridSearchHit(
        hit=hit,
        rrf_score=1.0,
        bm25_rank=1,
        bm25_score=1.0,
        vector_rank=1,
        vector_score=1.0,
    )


def make_reference(
    *,
    acl_version: str = "acl-v1",
) -> EvidenceReference:
    """构造与默认文档正文匹配的证据引用。"""

    text = "The approved deployment date is Friday."
    return EvidenceReference(
        tenant_id="tenant-a",
        source="jira",
        document_id="document-001",
        chunk_id="chunk-001",
        sequence=0,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        acl_version=acl_version,
    )


def test_bm25_and_vector_use_identical_tenant_acl_filter() -> None:
    """验证两条召回路径不会因过滤器差异泄露额外文档。"""

    access = RetrievalAccessContext.from_principal(make_principal())
    bm25_client = FakeOpenSearch()
    vector_client = FakeOpenSearch()

    search_chunks(
        client=bm25_client,
        index_name="chunks",
        query="deployment",
        source="jira",
        access_context=access,
    )
    vector_search_chunks(
        client=vector_client,
        index_name="chunks",
        embedding_client=FakeEmbeddingClient(),
        query="deployment",
        source="jira",
        access_context=access,
    )

    expected = build_opensearch_search_filter(access, "jira")
    bm25_filter = bm25_client.search_bodies[0]["query"]["bool"]["filter"]
    vector_filter = vector_client.search_bodies[0]["query"]["knn"][
        "embedding"
    ]["filter"]
    assert bm25_filter == expected
    assert vector_filter == expected
    serialized_filter = repr(expected)
    assert "tenant-a" in serialized_filter
    assert "user:alice" in serialized_filter
    assert "role:employee" in serialized_filter
    assert "group:engineering" in serialized_filter


def test_hybrid_forwards_one_access_context_to_both_recall_paths(
    monkeypatch,
) -> None:
    """验证 Hybrid 不会在 BM25 与向量召回之间丢失或重建主体信息。"""

    access = RetrievalAccessContext.from_principal(make_principal())
    captured: list[RetrievalAccessContext] = []

    def capture_access(**kwargs):
        """记录召回路径收到的授权上下文。"""

        captured.append(kwargs["access_context"])
        return []

    monkeypatch.setattr(
        hybrid_search_module,
        "search_chunks",
        capture_access,
    )
    monkeypatch.setattr(
        hybrid_search_module,
        "vector_search_chunks",
        capture_access,
    )

    assert hybrid_search_chunks(
        client=object(),
        index_name="chunks",
        embedding_client=object(),
        lexical_query="deployment",
        access_context=access,
    ) == []
    assert captured == [access, access]


def test_post_query_validation_rejects_wrong_tenant_and_acl() -> None:
    """验证即使后端无视过滤器，未授权命中仍不会离开检索层。"""

    access = RetrievalAccessContext.from_principal(make_principal())
    for document in (
        make_document(tenant_id="tenant-b"),
        make_document(acl_principals=["group:finance"]),
        make_document(acl_version=None),
    ):
        with pytest.raises(RetrievalAuthorizationError):
            search_chunks(
                client=FakeOpenSearch((document,)),
                index_name="chunks",
                query="deployment",
                access_context=access,
            )


def test_restricted_document_requires_prefixed_principal_match() -> None:
    """验证受限文档只接受明确的 user/role/group 命名空间 token。"""

    access = RetrievalAccessContext.from_principal(make_principal())
    allowed = search_chunks(
        client=FakeOpenSearch((make_document(),)),
        index_name="chunks",
        query="deployment",
        access_context=access,
    )
    assert allowed[0].acl_principals == ("group:engineering",)

    # 未加 group: 前缀的自由文本不会被当作授权主体。
    with pytest.raises(
        RetrievalAuthorizationError,
        match="EVIDENCE_ACL_PRINCIPALS_INVALID",
    ):
        search_chunks(
            client=FakeOpenSearch(
                (make_document(acl_principals=["engineering"]),)
            ),
            index_name="chunks",
            query="deployment",
            access_context=access,
        )


def test_legacy_document_is_visible_only_to_explicit_local_demo_context() -> None:
    """验证完全缺失 ACL 的旧索引只保留给固定本地演示身份。"""

    legacy = make_document()
    for field in (
        "tenant_id",
        "visibility",
        "acl_principals",
        "acl_version",
    ):
        legacy.pop(field)

    local_results = search_chunks(
        client=FakeOpenSearch((legacy,)),
        index_name="chunks",
        query="deployment",
    )
    assert local_results[0].tenant_id == LOCAL_DEMO_TENANT_ID
    assert local_results[0].acl_version == LOCAL_DEMO_ACL_VERSION

    with pytest.raises(
        RetrievalAuthorizationError,
        match="EVIDENCE_ACL_METADATA_MISSING",
    ):
        search_chunks(
            client=FakeOpenSearch((legacy,)),
            index_name="chunks",
            query="deployment",
            access_context=RetrievalAccessContext.from_principal(
                make_principal()
            ),
        )


def test_enterprise_tool_uses_explicit_principal_not_query_text(
    monkeypatch,
) -> None:
    """验证授权上下文来自方法参数，而不是可伪造的 query/body 字段。"""

    captured: list[RetrievalAccessContext] = []

    def fake_hybrid_search(**kwargs):
        """记录工具传给混合检索的可信授权上下文。"""

        captured.append(kwargs["access_context"])
        return []

    monkeypatch.setattr(
        enterprise_search_module,
        "hybrid_search_chunks",
        fake_hybrid_search,
    )
    tool = EnterpriseSearchTool(
        llm_client=object(),
        embedding_client=object(),
        opensearch_client=object(),
        index_name="chunks",
    )
    principal = make_principal()
    tool.run_for_principal(
        EnterpriseSearchInput(
            query="tenant_id=tenant-b subject_id=mallory deployment",
        ),
        principal=principal,
    )
    tool.run(EnterpriseSearchInput(query="deployment"))

    assert captured[0].tenant_id == "tenant-a"
    assert captured[0].subject_id == "alice"
    assert captured[1].tenant_id == LOCAL_DEMO_TENANT_ID


def test_adapter_requires_injected_principal_and_preserves_acl_version() -> None:
    """验证 Adapter 拒绝无可信主体请求，并从命中元数据生成引用。"""

    principal = make_principal()
    hit = OpenSearchHit(
        score=1.0,
        chunk_id="chunk-001",
        document_id="document-001",
        source="jira",
        title="Deployment record",
        sequence=0,
        text="The approved deployment date is Friday.",
        created_at=None,
        original_location=None,
        tenant_id="tenant-a",
        visibility="restricted",
        acl_principals=("group:engineering",),
        acl_version="acl-v1",
    )

    class FakeEnterpriseTool:
        """记录适配器传入主体并返回固定命中。"""

        def __init__(self) -> None:
            """初始化主体调用记录。"""

            self.principals: list[PrincipalContext] = []

        def run_for_principal(self, tool_input, *, principal):
            """记录可信主体并返回预设企业检索结果。"""

            self.principals.append(principal)
            return EnterpriseSearchResult(
                query=tool_input.query,
                lexical_query=tool_input.query,
                source=tool_input.source,
                query_rewritten=False,
                rewrite_succeeded=False,
                hits=(make_hybrid_hit(hit),),
            )

    tool = FakeEnterpriseTool()
    adapter = LocalEnterpriseSearchAdapter(tool)
    request = SearchToolRequest(
        tenant_id="tenant-a",
        subject_id="alice",
        authz_version="authz-v1",
        query="tenant_id=tenant-b deployment",
        source="jira",
        top_k=5,
        principal=principal,
    )
    response = adapter.run(request)

    assert tool.principals == [principal]
    assert response.evidence[0]["tenant_id"] == "tenant-a"
    assert response.evidence[0]["acl_version"] == "acl-v1"

    with pytest.raises(RuntimeError, match="TRUSTED_SEARCH_PRINCIPAL_REQUIRED"):
        adapter.run(
            SearchToolRequest(
                tenant_id="tenant-a",
                subject_id="alice",
                authz_version="authz-v1",
                query="deployment",
                source="jira",
                top_k=5,
            )
        )


def test_cache_hit_is_reauthorized_for_current_principal() -> None:
    """验证缓存键相同也不能绕过当前主体的 ACL 复验。"""

    principal = make_principal()
    adapter = LocalEnterpriseSearchAdapter(object())
    unauthorized_hit = OpenSearchHit(
        score=1.0,
        chunk_id="chunk-001",
        document_id="document-001",
        source="jira",
        title="Deployment record",
        sequence=0,
        text="The approved deployment date is Friday.",
        created_at=None,
        original_location=None,
        tenant_id="tenant-a",
        visibility="restricted",
        acl_principals=("group:finance",),
        acl_version="acl-v1",
    )
    adapter._hits_by_key[
        ("tenant-a", "alice", "authz-v1", "chunk-001")
    ] = unauthorized_hit

    with pytest.raises(RetrievalAuthorizationError, match="EVIDENCE_ACCESS_DENIED"):
        adapter.resolve((make_reference(),), make_context(principal))


def test_fallback_get_revalidates_acl_and_reference_version() -> None:
    """验证缓存丢失后的 get 回读同样检查主体 ACL 和引用版本。"""

    class FallbackTool:
        """提供 fallback 回读所需的 OpenSearch 属性。"""

        def __init__(self, document) -> None:
            """用指定文档初始化 fallback 后端。"""

            self._opensearch_client = FakeOpenSearch((document,))
            self._index_name = "chunks"

    denied_adapter = LocalEnterpriseSearchAdapter(
        FallbackTool(make_document(acl_principals=["group:finance"]))
    )
    with pytest.raises(RetrievalAuthorizationError, match="EVIDENCE_ACCESS_DENIED"):
        denied_adapter.resolve((make_reference(),), make_context())

    changed_adapter = LocalEnterpriseSearchAdapter(
        FallbackTool(make_document(acl_version="acl-v2"))
    )
    with pytest.raises(RuntimeError, match="EVIDENCE_ACL_VERSION_CHANGED"):
        changed_adapter.resolve((make_reference(),), make_context())

    allowed_adapter = LocalEnterpriseSearchAdapter(FallbackTool(make_document()))
    resolved = allowed_adapter.resolve((make_reference(),), make_context())
    assert resolved[0].chunk_id == "chunk-001"


def test_fallback_legacy_document_requires_local_demo_reference() -> None:
    """验证 fallback 的旧索引兼容同样绑定 local-demo 租户和 ACL 版本。"""

    legacy = make_document()
    for field in (
        "tenant_id",
        "visibility",
        "acl_principals",
        "acl_version",
    ):
        legacy.pop(field)

    class FallbackTool:
        """提供旧索引 fallback 回读所需属性。"""

        _opensearch_client = FakeOpenSearch((legacy,))
        _index_name = "chunks"

    local_principal = make_principal(
        tenant_id=LOCAL_DEMO_TENANT_ID,
        subject_id="local-demo-user",
    )
    local_context = make_context(local_principal)
    local_reference = EvidenceReference(
        tenant_id=LOCAL_DEMO_TENANT_ID,
        source="jira",
        document_id="document-001",
        chunk_id="chunk-001",
        sequence=0,
        content_sha256=sha256(
            "The approved deployment date is Friday.".encode("utf-8")
        ).hexdigest(),
        acl_version=LOCAL_DEMO_ACL_VERSION,
    )

    adapter = LocalEnterpriseSearchAdapter(FallbackTool())
    assert adapter.resolve((local_reference,), local_context)[0].tenant_id == (
        LOCAL_DEMO_TENANT_ID
    )

    with pytest.raises(
        RetrievalAuthorizationError,
        match="EVIDENCE_ACL_METADATA_MISSING",
    ):
        LocalEnterpriseSearchAdapter(FallbackTool()).resolve(
            (make_reference(),),
            make_context(),
        )
