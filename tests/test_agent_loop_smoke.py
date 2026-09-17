"""在不访问外部服务的前提下冒烟验证两轮 Agent 循环。

两步伪规划器、检索器和审核器共同模拟“首次证据不足、补充检索后可回答”的完整路径。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

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
from enterprise_agent.infrastructure.llm_client import (
    LLMResult,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
)


class TwoStepPlanner:
    """连续提出两条互补搜索查询的两轮规划器替身。"""

    def __init__(self) -> None:
        """将规划轮次计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """按轮次返回归档流程或变更单查询，超出两轮则失败。"""
        queries = [
            "project lifecycle archival process",
            (
                "approved change ticket "
                "deployment date"
            ),
        ]

        if self.calls >= len(queries):
            raise AssertionError(
                "Planner was called too many times"
            )

        query = queries[self.calls]
        self.calls += 1

        return LLMResult(
            text=json.dumps(
                {
                    "kind": (
                        "search_enterprise_knowledge"
                    ),
                    "query": query,
                    "source": "jira",
                    "top_k": 1,
                    "reason": (
                        "Search for the next "
                        "required evidence."
                    ),
                }
            ),
            model="fake-planner",
            request_id=f"planner-{self.calls}",
            input_tokens=10,
            output_tokens=10,
        )


class TwoStepSearchTool:
    """每轮生成一条新证据引用并保存请求的搜索工具替身。"""

    def __init__(self) -> None:
        """初始化空的搜索请求历史。"""
        self.requests: list[SearchToolRequest] = []

    def run(
        self,
        request: SearchToolRequest,
    ) -> SearchToolResponse:
        """记录请求，并依据轮次生成唯一的证据引用。"""
        self.requests.append(request)

        number = len(self.requests)

        reference = EvidenceReference(
            tenant_id=request.tenant_id,
            source="jira",
            document_id=f"document-{number}",
            chunk_id=f"chunk-{number}",
            sequence=number - 1,
            content_sha256=(
                chr(96 + number) * 64
            ),
            acl_version="acl-v1",
        )

        return SearchToolResponse(
            evidence=(reference,),
        )


class FakeEvidenceResolver:
    """按块编号把两轮证据引用解析成不同正文。"""

    def __init__(self) -> None:
        """将正文解析调用计数初始化为零。"""
        self.calls = 0

    def resolve(
        self,
        references: Sequence[EvidenceReference],
        runtime_context: AgentRuntimeContext,
    ) -> Sequence[OpenSearchHit]:
        """将引用解析为流程正文或包含变更单编号的正文。"""
        self.calls += 1

        return tuple(
            OpenSearchHit(
                score=1.0,
                chunk_id=reference["chunk_id"],
                document_id=reference["document_id"],
                source=reference["source"],
                title=(
                    "Project archival evidence"
                ),
                sequence=reference["sequence"],
                text=(
                    "chunk-1"
                    if reference["chunk_id"]
                    == "chunk-1"
                    else (
                        "Approved change ticket "
                        "is CHG-2026-001."
                    )
                ),
                created_at=None,
                original_location=None,
            )
            for reference in references
        )


class TwoStepJudge:
    """首轮返回部分支持、次轮返回完全支持的审核器替身。"""

    def __init__(self) -> None:
        """将审核轮次计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """按轮次返回部分支持或完全支持，第三次调用则失败。"""
        self.calls += 1

        if self.calls == 1:
            payload = {
                "decision": "partially_supported",
                "facts": [
                    {
                        "fact": "archival process",
                        "status": "supported",
                        "conclusion": (
                            "The process was found."
                        ),
                        "evidence_ids": ["E1"],
                    },
                    {
                        "fact": (
                            "approved change-ticket ID"
                        ),
                        "status": "missing",
                        "conclusion": None,
                        "evidence_ids": [],
                    },
                ],
                "reason": (
                    "The process was found, "
                    "but the ticket ID is missing."
                ),
            }

            input_tokens = 100
            output_tokens = 30

        elif self.calls == 2:
            payload = {
                "decision": "supported",
                "facts": [
                    {
                        "fact": "archival process",
                        "status": "supported",
                        "conclusion": (
                            "The process was found."
                        ),
                        "evidence_ids": ["E1"],
                    },
                    {
                        "fact": (
                            "approved change-ticket ID"
                        ),
                        "status": "supported",
                        "conclusion": "CHG-2026-001",
                        "evidence_ids": ["E2"],
                    },
                ],
                "reason": (
                    "All required facts are now "
                    "supported."
                ),
            }

            input_tokens = 120
            output_tokens = 40

        else:
            raise AssertionError(
                "Judge was called too many times"
            )

        return LLMResult(
            text=json.dumps(payload),
            model="fake-judge",
            request_id=f"judge-{self.calls}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def make_context() -> AgentRuntimeContext:
    """构造两轮循环冒烟测试使用的固定运行上下文。"""
    return AgentRuntimeContext(
        request_id="loop-smoke-001",
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


def test_two_round_agent_loop() -> None:
    """验证缺失事实驱动第二轮检索并最终得到完全支持结论。"""
    context = make_context()

    state = create_initial_graph_state(
        goal=(
            "Find the project archival process "
            "and approved change-ticket ID."
        ),
        runtime_context=context,
    )

    planner = TwoStepPlanner()
    search_tool = TwoStepSearchTool()
    resolver = FakeEvidenceResolver()
    judge = TwoStepJudge()

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

    print("\n========== 循环结果 ==========")
    print("最终状态：", result["status"])
    print("Planner调用：", planner.calls)
    print(
        "搜索次数：",
        len(search_tool.requests),
    )
    print("Resolver调用：", resolver.calls)
    print("Judge调用：", judge.calls)
    print("规划步数：", result["step_count"])
    print(
        "证据数量：",
        len(result["evidence_refs"]),
    )
    print(
        "最终审核：",
        result["evidence_assessment"]["decision"],
    )
    print(
        "模型Token：",
        result["model_token_count"],
    )

    assert result["status"] == "completed"

    assert planner.calls == 2
    assert len(search_tool.requests) == 2
    assert resolver.calls == 2
    assert judge.calls == 2

    assert result["step_count"] == 2
    assert result["tool_call_count"] == 2
    assert len(result["evidence_refs"]) == 2

    assessment = result["evidence_assessment"]

    assert assessment is not None
    assert assessment["decision"] == "supported"

    assert result["model_token_count"] == 330

    assert search_tool.requests[0].query == (
        "project lifecycle archival process"
    )
    assert search_tool.requests[1].query == (
        "approved change ticket deployment date"
    )
