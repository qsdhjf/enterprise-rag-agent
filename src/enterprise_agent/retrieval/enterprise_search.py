"""面向 Agent 的只读企业搜索工具：校验输入、改写查询并执行混合检索。"""

from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import OpenSearch

from enterprise_agent.agent.contracts import PrincipalContext
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.infrastructure.reranker_client import (
    TEIRerankerClient,
)
from enterprise_agent.retrieval.hybrid_search import (
    HybridSearchHit,
    hybrid_search_chunks,
)
from enterprise_agent.retrieval.access_control import (
    RetrievalAccessContext,
)
from enterprise_agent.retrieval.query_rewriter import rewrite_query


# 白名单既帮助尽早发现拼写错误，也防止把任意值传入来源过滤条件。
SUPPORTED_SOURCES = frozenset(
    {
        "confluence",
        "fireflies",
        "github",
        "gmail",
        "google_drive",
        "hubspot",
        "jira",
        "linear",
        "slack",
    }
)


@dataclass(frozen=True, slots=True)
class EnterpriseSearchInput:
    """企业搜索工具的查询、可选数据源和返回条数。"""

    query: str
    source: str | None = None
    top_k: int = 5


@dataclass(frozen=True, slots=True)
class EnterpriseSearchResult:
    """一次搜索的实际检索词、改写状态及不可变命中结果。"""

    query: str
    lexical_query: str
    source: str | None

    query_rewritten: bool
    rewrite_succeeded: bool

    hits: tuple[HybridSearchHit, ...]


class EnterpriseSearchTool:
    """协调 LLM 查询改写、Embedding 和 OpenSearch 混合检索的工具对象。"""

    name = "search_enterprise_knowledge"

    description = (
        "只读搜索企业知识库。支持跨来源搜索，"
        "也可以限定jira、slack、confluence等来源。"
        "返回带文档ID和切块ID的企业证据，不执行写操作。"
    )

    def __init__(
        self,
        llm_client: OpenAICompatibleLLMClient,
        embedding_client: OpenAICompatibleEmbeddingClient,
        opensearch_client: OpenSearch,
        index_name: str,
        reranker_client: TEIRerankerClient | None = None,
    ) -> None:
        """注入外部服务客户端和目标混合索引名。"""

        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self._opensearch_client = opensearch_client
        self._index_name = index_name
        self._reranker_client = reranker_client

    def run(
        self,
        tool_input: EnterpriseSearchInput,
    ) -> EnterpriseSearchResult:
        """使用固定本地演示身份执行旧 CLI/API 调用。"""

        return self._run(
            tool_input,
            access_context=RetrievalAccessContext.local_demo(),
        )

    def run_for_principal(
        self,
        tool_input: EnterpriseSearchInput,
        *,
        principal: PrincipalContext,
    ) -> EnterpriseSearchResult:
        """使用服务端注入的可信主体执行租户和 ACL 隔离检索。"""

        return self._run(
            tool_input,
            access_context=(
                RetrievalAccessContext.from_principal(principal)
            ),
        )

    def _run(
        self,
        tool_input: EnterpriseSearchInput,
        *,
        access_context: RetrievalAccessContext,
    ) -> EnterpriseSearchResult:
        """校验工具输入，并返回带改写元数据的混合搜索结果。

        查询改写属于可降级增强：模型调用失败时会继续用原查询搜索；
        真正的检索异常则继续向上抛出，避免把服务故障伪装成“没有结果”。
        """

        query = tool_input.query.strip()

        if not query:
            raise ValueError("搜索词不能为空")

        if not 1 <= tool_input.top_k <= 10:
            raise ValueError(
                "top_k必须在1到10之间"
            )

        # 来源统一转成小写，便于与索引中的 keyword 值和白名单比较。
        source = (
            tool_input.source.strip().lower()
            if tool_input.source
            else None
        )

        if (
            source is not None
            and source not in SUPPORTED_SOURCES
        ):
            raise ValueError(
                f"不支持的数据源：{source}"
            )

        # 中文问题转换成适合英文企业文档的检索词。
        # 改写失败时仍然使用原搜索词继续工作。
        try:
            rewrite_result = rewrite_query(
                query=query,
                client=self._llm_client,
            )
            lexical_query = (
                rewrite_result.retrieval_query
            )
            query_rewritten = (
                rewrite_result.rewritten
            )
            rewrite_succeeded = True
        # 改写不是检索的硬依赖，任何模型/网络错误都回退到原始搜索词。
        except Exception:
            lexical_query = query
            query_rewritten = False
            rewrite_succeeded = False

        hits = hybrid_search_chunks(
            client=self._opensearch_client,
            index_name=self._index_name,
            embedding_client=self._embedding_client,

            # BM25 使用原问题与英文改写的组合词。
            lexical_query=lexical_query,

            # 向量模型使用原始语义问题，避免翻译改变用户意图。
            semantic_query=query,

            top_k=tool_input.top_k,
            source=source,
            max_chunks_per_document=2,
            reranker_client=self._reranker_client,
            access_context=access_context,
        )

        return EnterpriseSearchResult(
            query=query,
            lexical_query=lexical_query,
            source=source,
            query_rewritten=query_rewritten,
            rewrite_succeeded=rewrite_succeeded,
            hits=tuple(hits),
        )
