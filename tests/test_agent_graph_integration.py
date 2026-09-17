"""用可控的伪依赖验证完整 LangGraph 工作流。

这里关注节点之间的状态传递，以及过期、规划失败和空检索结果能否安全地提前终止。
"""

import json
from datetime import datetime, timedelta, timezone
from collections.abc import Sequence
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
)
from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.graph import (
    build_agent_graph,
)
from enterprise_agent.agent.state import (
    EvidenceReference,
    create_initial_graph_state,
)
from enterprise_agent.agent.tool_executor import (
    SearchToolRequest,
    SearchToolResponse,
)
from enterprise_agent.infrastructure.llm_client import LLMResult


class FakePlannerClient:
    """模拟 Planner 大模型。"""

    def __init__(self) -> None:
        """将规划器调用计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """返回一次搜索 Jira 的固定规划提案并记录调用。"""
        self.calls += 1

        return LLMResult(
            text=json.dumps(
                {
                    "kind": "search_enterprise_knowledge",
                    "query": "project lifecycle archival",
                    "source": "jira",
                    "top_k": 10,
                    "reason": "Need enterprise evidence.",
                }
            ),
            model="fake-planner",
            request_id="fake-request-001",
            input_tokens=100,
            output_tokens=50,
        )


class BrokenPlannerClient:
    """模拟 Planner 模型调用失败。"""

    def generate(self, **kwargs):
        """抛出模拟异常以验证规划失败后的图终止行为。"""
        raise RuntimeError("Simulated planner failure")


class FakeSearchTool:
    """模拟企业知识搜索工具。"""

    def __init__(
        self,
        evidence: tuple[EvidenceReference, ...],
    ) -> None:
        """保存预设证据并初始化搜索请求记录。"""
        self.evidence = evidence
        self.requests: list[SearchToolRequest] = []

    def run(
        self,
        request: SearchToolRequest,
    ) -> SearchToolResponse:
        """记录实际执行的获批请求并返回预设证据。"""
        self.requests.append(request)

        return SearchToolResponse(
            evidence=self.evidence,
        )

class FakeEvidenceResolver:
    """把证据引用转换成固定项目归档正文的解析器替身。"""

    def __init__(self) -> None:
        """将证据解析调用计数初始化为零。"""
        self.calls = 0

    def resolve(
        self,
        references: Sequence[EvidenceReference],
        runtime_context: AgentRuntimeContext,
    ) -> Sequence[OpenSearchHit]:
        """为每条引用构造匹配的检索命中并记录调用。"""
        self.calls += 1

        return tuple(
            OpenSearchHit(
                score=1.0,
                chunk_id=reference["chunk_id"],
                document_id=reference["document_id"],
                source=reference["source"],
                title="Project archival process",
                sequence=reference["sequence"],
                text=(
                    "The design requires closing "
                    "Slack channels, snapshotting "
                    "Confluence pages, and archiving "
                    "related Jira issues."
                ),
                created_at=None,
                original_location=None,
            )
            for reference in references
        )


class FakeJudgeClient:
    """返回证据完全支持结论的假审核模型。"""

    def __init__(self) -> None:
        """将审核调用计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """返回合法 supported 审核 JSON 并记录固定 Token 用量。"""
        self.calls += 1

        return LLMResult(
            text=json.dumps(
                {
                    "decision": "supported",
                    "facts": [
                        {
                            "fact": (
                                "project archival process"
                            ),
                            "status": "supported",
                            "conclusion": (
                                "The evidence describes "
                                "the archival process."
                            ),
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "reason": (
                        "The required process is "
                        "supported by the evidence."
                    ),
                }
            ),
            model="fake-judge",
            request_id="fake-judge-001",
            input_tokens=200,
            output_tokens=50,
        )


class FakeAnswerClient:
    """生成带有效 E1 引用的固定最终回答。"""

    def __init__(self) -> None:
        """将最终回答模型调用计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """返回带 E1 引用的固定回答并记录调用次数。"""
        self.calls += 1

        return LLMResult(
            text=(
                "The project follows the documented "
                "archival process [E1]."
            ),
            model="fake-answer",
            request_id="fake-answer-001",
            input_tokens=100,
            output_tokens=20,
        )

def make_context() -> AgentRuntimeContext:
    """构造完整图集成测试使用的运行身份、预算和检索策略。"""
    return AgentRuntimeContext(
        request_id="graph-test-001",
        principal=PrincipalContext(
            tenant_id="tenant-a",
            subject_id="alice",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="authz-v1",
        ),
        budget=RunBudget(),
        retrieval_policy=RetrievalPolicy(
            policy_version="policy-v1",
            allowed_sources=frozenset({"jira"}),
            allow_cross_source=False,
            max_top_k=5,
        ),
    )


def make_evidence() -> tuple[
    EvidenceReference,
    ...,
]:
    """构造两条属于当前租户的固定证据引用。"""
    return (
        EvidenceReference(
            tenant_id="tenant-a",
            source="jira",
            document_id="document-001",
            chunk_id="chunk-001",
            sequence=0,
            content_sha256="a" * 64,
            acl_version="acl-v1",
        ),
        EvidenceReference(
            tenant_id="tenant-a",
            source="jira",
            document_id="document-002",
            chunk_id="chunk-002",
            sequence=0,
            content_sha256="b" * 64,
            acl_version="acl-v1",
        ),
    )


def test_full_graph_executes_approved_search() -> None:
    """验证完整图采用获批搜索并形成带引用的受支持回答。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="Investigate project archival workflow",
        runtime_context=context,
    )

    planner = FakePlannerClient()
    search_tool = FakeSearchTool(make_evidence())
    resolver = FakeEvidenceResolver()
    judge = FakeJudgeClient()
    answer = FakeAnswerClient()

    graph = build_agent_graph(
        planner,
        search_tool,
        resolver,
        judge,
        answer,
    )

    result = graph.invoke(
        state,
        context=context,
    )

    # Planner只调用了一次。
    assert planner.calls == 1

    # 搜索工具也只调用了一次。
    assert len(search_tool.requests) == 1

    executed_request = search_tool.requests[0]

    # 模型建议10条，但执行的是系统批准后的5条。
    assert executed_request.top_k == 5
    assert executed_request.source == "jira"
    assert executed_request.tenant_id == "tenant-a"
    assert executed_request.subject_id == "alice"
    assert executed_request.principal is context.principal

    # Agent运行状态发生了变化。
    assert result["status"] == "completed"
    assert result["step_count"] == 1
    assert result["tool_call_count"] == 1
    assert result["retrieved_chunk_count"] == 2

    # 动作已经执行完，不能继续留在待执行区域。
    assert result["pending_action"] is None
    assert result["approved_action"] is None

    # 两条证据引用已进入Agent状态。
    assert len(result["evidence_refs"]) == 2

    # 工具调用审计记录已经保存。
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["status"] == "succeeded"
    assert result["tool_calls"][0]["returned_hits"] == 2
    # 审核和最终回答分别重新加载并校验证据一次。
    assert resolver.calls == 2
    assert judge.calls == 1

    assessment = result["evidence_assessment"]

    assert assessment is not None
    assert assessment["decision"] == "supported"
    assert assessment["judge_succeeded"] is True
    assert assessment["used_model"] is True

    assert assessment["assessed_chunk_ids"] == [
        "chunk-001",
        "chunk-002",
    ]

    final_response = result["final_response"]

    assert final_response is not None
    assert final_response["outcome"] == "supported"
    assert final_response["citations"] == ["E1"]
    assert answer.calls == 1

    # Planner使用150、Judge使用250、Answer使用120 Token。
    assert result["model_token_count"] == 520

def test_expired_task_does_not_call_dependencies() -> None:
    """验证任务过期时图在调用规划器、工具或审核器前停止。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="Investigate project archival workflow",
        runtime_context=context,
    )

    state["deadline_at"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=1)
    ).isoformat()

    planner = FakePlannerClient()
    search_tool = FakeSearchTool(make_evidence())
    resolver = FakeEvidenceResolver()
    judge = FakeJudgeClient()
    graph = build_agent_graph(
        planner,
        search_tool,
        resolver,
        judge,
    )

    result = graph.invoke(
        state,
        context=context,
    )

    # 任务已经过期，模型和工具都不允许调用。
    assert planner.calls == 0
    assert len(search_tool.requests) == 0

    assert result["status"] == "budget_exceeded"
    assert (
        result["last_error_code"]
        == "DEADLINE_EXCEEDED"
    )


def test_planner_failure_does_not_call_search() -> None:
    """验证规划器故障后图标记失败且不调用后续搜索与审核依赖。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="Investigate project archival workflow",
        runtime_context=context,
    )

    search_tool = FakeSearchTool(make_evidence())
    resolver = FakeEvidenceResolver()
    judge = FakeJudgeClient()
    graph = build_agent_graph(
        BrokenPlannerClient(),
        search_tool,
        resolver,
        judge,
    )
    result = graph.invoke(
        state,
        context=context,
    )

    # Planner失败后，图必须停止，不能错误调用工具。
    assert len(search_tool.requests) == 0
    assert resolver.calls == 0
    assert judge.calls == 0
    assert result["status"] == "failed"
    assert result["pending_action"] is None
    assert result["approved_action"] is None
    assert (
        result["last_error_code"]
        == "PLANNER_MODEL_FAILED"
    )

def test_empty_search_becomes_unsupported() -> None:
    """验证空搜索结果形成无需审核模型的 unsupported 最终响应。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="What is the exact deployment date?",
        runtime_context=context,
    )

    planner = FakePlannerClient()

    # 搜索工具正常运行，但没有找到任何内容。
    search_tool = FakeSearchTool(
        evidence=(),
    )

    resolver = FakeEvidenceResolver()
    judge = FakeJudgeClient()
    answer = FakeAnswerClient()

    graph = build_agent_graph(
        planner,
        search_tool,
        resolver,
        judge,
        answer,
    )

    result = graph.invoke(
        state,
        context=context,
    )

    assert result["status"] == "completed"

    # 搜索工具确实执行了一次，但返回0条。
    assert result["tool_call_count"] == 1
    assert result["retrieved_chunk_count"] == 0
    assert result["evidence_refs"] == []

    assessment = result["evidence_assessment"]

    assert assessment is not None
    assert assessment["decision"] == "unsupported"
    assert assessment["judge_succeeded"] is True
    assert assessment["used_model"] is False
    assert assessment["assessed_chunk_ids"] == []

    # 没有证据正文，所以Resolver和审核模型都不调用。
    assert resolver.calls == 0
    assert judge.calls == 0

    final_response = result["final_response"]

    assert final_response is not None
    assert final_response["outcome"] == "unsupported"
    assert answer.calls == 0

    # Planner尝试了两次；拒答本身不调用答案模型。
    assert result["model_token_count"] == 300
