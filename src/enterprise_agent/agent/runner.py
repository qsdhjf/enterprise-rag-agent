"""提供面向 API/服务层的一次性 Agent 运行入口与依赖就绪检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from opensearchpy import OpenSearch

from enterprise_agent.agent.contracts import AgentRuntimeContext
from enterprise_agent.agent.presentation import Activity, EvidenceCard, FactCard
from enterprise_agent.agent.search_adapter import (
    LocalEnterpriseSearchAdapter,
)
from enterprise_agent.agent.graph import build_agent_graph
from enterprise_agent.agent.state import (
    EvidenceDecision,
    create_initial_graph_state,
)
from enterprise_agent.agent.scope_gate import (
    ScopeCategory,
    user_message_for_scope,
)
from enterprise_agent.agent.scope_service import (
    evaluate_request_scope,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.retrieval.enterprise_search import (
    EnterpriseSearchTool,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.infrastructure.reranker_client import (
    TEIRerankerClient,
)
from enterprise_agent.generation.resilient_judge_client import (
    RetryingEvidenceJudgeClient,
)


AgentRunStatus = Literal[
    "completed",
    "scope_rejected",
]


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """一次用户请求的稳定输出，包括回答、引用和资源用量摘要。"""

    status: AgentRunStatus
    request_id: str
    task_id: str | None

    question: str
    scope_category: ScopeCategory
    outcome: EvidenceDecision | None
    answer: str
    citations: tuple[str, ...]

    step_count: int
    tool_call_count: int
    evidence_count: int
    model_token_count: int
    evidence: tuple[EvidenceCard, ...] = ()
    facts: tuple[FactCard, ...] = ()
    activity: tuple[Activity, ...] = ()
    mode: str = "live"

@dataclass(frozen=True, slots=True)
class AgentReadiness:
    """OpenSearch、索引和数据是否足以启动一次 Agent 调查。"""

    ready: bool
    opensearch_reachable: bool
    hybrid_index_exists: bool
    indexed_chunks: int | None
    message: str
    reranker_ready: bool | None = None
    reranker_model: str | None = None


class AgentRunnerUnavailable(RuntimeError):
    """表示运行环境或图执行未能安全地产生最终回答。"""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        """保存机器可读错误码，同时保留普通异常消息。"""

        super().__init__(message)
        self.error_code = error_code


class AgentRunner:
    """运行一次完整的 Scope Gate + LangGraph 企业知识 Agent。"""

    def __init__(
        self,
        llm_client: OpenAICompatibleLLMClient,
        embedding_client: OpenAICompatibleEmbeddingClient,
        opensearch_client: OpenSearch,
        hybrid_index_name: str,
        reranker_client: TEIRerankerClient | None = None,
        include_evidence: bool = False,
    ) -> None:
        """注入模型、向量、OpenSearch 客户端和目标混合索引。"""

        self._llm_client = llm_client
        self._include_evidence = include_evidence
        self._opensearch_client = opensearch_client
        self._hybrid_index_name = hybrid_index_name

        self._reranker_client = reranker_client
        # 搜索工具本身不保存某次请求的证据正文，可以进程级复用。
        self._search_tool = EnterpriseSearchTool(
            llm_client=llm_client,
            embedding_client=embedding_client,
            opensearch_client=opensearch_client,
            index_name=hybrid_index_name,
            reranker_client=reranker_client,
        )

    def check_readiness(self) -> AgentReadiness:
        """以不抛出依赖异常的方式检查 OpenSearch 与目标索引。"""

        try:
            reachable = bool(
                self._opensearch_client.ping()
            )

            if not reachable:
                return AgentReadiness(
                    ready=False,
                    opensearch_reachable=False,
                    hybrid_index_exists=False,
                    indexed_chunks=None,
                    message="OpenSearch 不可访问",
                )

            index_exists = bool(
                self._opensearch_client.indices.exists(
                    index=self._hybrid_index_name,
                )
            )

            if not index_exists:
                return AgentReadiness(
                    ready=False,
                    opensearch_reachable=True,
                    hybrid_index_exists=False,
                    indexed_chunks=None,
                    message="混合检索索引不存在",
                )

            indexed_chunks = int(
                self._opensearch_client.count(
                    index=self._hybrid_index_name,
                )["count"]
            )

            if indexed_chunks == 0:
                return AgentReadiness(
                    ready=False,
                    opensearch_reachable=True,
                    hybrid_index_exists=True,
                    indexed_chunks=0,
                    message="混合检索索引中没有切块",
                )

            reranker_ready: bool | None = None
            if self._reranker_client is not None:
                reranker_ready = self._reranker_client.is_ready()

                if (
                    self._reranker_client.required
                    and not reranker_ready
                ):
                    return AgentReadiness(
                        ready=False,
                        opensearch_reachable=True,
                        hybrid_index_exists=True,
                        indexed_chunks=indexed_chunks,
                        message="Reranker service is not ready",
                        reranker_ready=False,
                        reranker_model=(
                            self._reranker_client.model
                        ),
                    )

            return AgentReadiness(
                ready=True,
                opensearch_reachable=True,
                hybrid_index_exists=True,
                indexed_chunks=indexed_chunks,
                message="LangGraph Agent 依赖已就绪",
                reranker_ready=reranker_ready,
                reranker_model=(
                    self._reranker_client.model
                    if self._reranker_client is not None
                    else None
                ),
            )
        except Exception:
            return AgentReadiness(
                ready=False,
                opensearch_reachable=False,
                hybrid_index_exists=False,
                indexed_chunks=None,
                message="就绪检查执行失败",
            )

    def ask(
        self,
        question: str,
        runtime_context: AgentRuntimeContext,
    ) -> AgentRunResult:
        """校验问题、执行范围门禁，再运行状态图并整理公开结果。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("question 不能为空")

        if len(cleaned_question) > 2000:
            raise ValueError(
                "question 最多允许 2000 个字符"
            )

        # 范围门禁先于规划和检索，越界/写操作请求不会触碰企业数据源。
        scope_result = evaluate_request_scope(
            cleaned_question,
            self._llm_client,
        )
        # Scope Gate 在图外运行，其 token 需在最终统计中单独加回。
        scope_tokens = (
            scope_result.input_tokens
            + scope_result.output_tokens
        )

        if (
            not scope_result.classifier_succeeded
            or scope_result.category is None
        ):
            raise AgentRunnerUnavailable(
                "SCOPE_CLASSIFIER_FAILED",
                "系统暂时无法判断请求范围",
            )

        # 合法但非企业检索类的请求直接返回固定用户提示，不创建 task_id。
        if not scope_result.allowed:
            return AgentRunResult(
                status="scope_rejected",
                request_id=runtime_context.request_id,
                task_id=None,
                question=cleaned_question,
                scope_category=scope_result.category,
                outcome=None,
                answer=user_message_for_scope(
                    scope_result.category
                ),
                citations=(),
                step_count=0,
                tool_call_count=0,
                evidence_count=0,
                model_token_count=scope_tokens,
            )

        initial_state = create_initial_graph_state(
            goal=cleaned_question,
            runtime_context=runtime_context,
        )

        # 这些对象含有请求级可变状态，不能跨请求共享。
        adapter = LocalEnterpriseSearchAdapter(
            self._search_tool
        )
        judge_client = RetryingEvidenceJudgeClient(
            self._llm_client
        )
        graph = build_agent_graph(
            planner_client=self._llm_client,
            search_tool=adapter,
            evidence_resolver=adapter,
            judge_client=judge_client,
            answer_client=self._llm_client,
        )

        # runtime_context 通过 LangGraph context 注入，不由不可信状态覆盖。
        final_state = graph.invoke(
            initial_state,
            context=runtime_context,
        )
        final_response = final_state["final_response"]

        if (
            final_state["status"] != "completed"
            or final_response is None
        ):
            raise AgentRunnerUnavailable(
                final_state["last_error_code"]
                or "AGENT_RUN_INCOMPLETE",
                "Agent 未能安全完成本次请求",
            )

        cards: tuple[EvidenceCard, ...] = ()
        if self._include_evidence:
            try:
                # Resolve in the SAME order as the answer prompt's E1, E2, ... .
                # Read current index ACLs again: revoked/deleted evidence fails closed.
                hits = adapter.resolve(final_state["evidence_refs"], runtime_context, fresh=True)
                cards = tuple(EvidenceCard(
                    id=f"E{i}", source=hit.source, document_id=hit.document_id,
                    chunk_id=hit.chunk_id, title=hit.title, excerpt=hit.text[:4000],
                    url=hit.original_location,
                ) for i, hit in enumerate(hits, 1))
            except Exception as error:
                raise AgentRunnerUnavailable("EVIDENCE_EXPORT_FAILED", "证据已变更或不可访问") from error
        assessment = final_state.get("evidence_assessment")
        facts = tuple(FactCard(**fact) for fact in assessment["facts"]) if assessment else ()
        activity = tuple(Activity(
            label=call["tool_name"], status=call["status"],
            detail=f"{call['query']} · {call['returned_hits']} hits",
        ) for call in final_state.get("tool_calls", []))

        return AgentRunResult(
            status="completed",
            evidence=cards, facts=facts, activity=activity,
            request_id=runtime_context.request_id,
            task_id=final_state["task_id"],
            question=cleaned_question,
            scope_category=scope_result.category,
            outcome=final_response["outcome"],
            answer=final_response["answer"],
            citations=tuple(
                final_response["citations"]
            ),
            step_count=final_state["step_count"],
            tool_call_count=(
                final_state["tool_call_count"]
            ),
            evidence_count=len(
                final_state["evidence_refs"]
            ),
            model_token_count=(
                final_state["model_token_count"]
                + scope_tokens
            ),
        )
