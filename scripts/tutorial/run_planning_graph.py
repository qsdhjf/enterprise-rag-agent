"""规划图教程：用假搜索工具观察 Planner、策略门和工具节点的协作。"""

from typing import get_args

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
    SourceName,
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
from enterprise_agent.infrastructure.config import load_llm_settings
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)


class DemoSearchTool:
    """不访问 OpenSearch、始终返回模拟证据的教学用搜索工具。"""

    def __init__(self) -> None:
        """创建工具并准备一个列表，用于记录图实际发出的请求。"""
        self.requests: list[SearchToolRequest] = []

    def run(
        self,
        request: SearchToolRequest,
    ) -> SearchToolResponse:
        """记录结构化请求，并返回两条满足工具协议的证据引用。"""
        # 保存请求可让教程在图运行后比较“模型建议”和“实际调用”。
        self.requests.append(request)

        # 如果 Planner 没有限定来源，演示工具默认返回 Jira 证据。
        evidence_source = request.source or "jira"

        # 两条引用使用固定 ID，便于多次运行时比较状态变化。
        evidence = (
            EvidenceReference(
                tenant_id=request.tenant_id,
                source=evidence_source,
                document_id=(
                    "dsid_eab08bf08daa480cae4e2f511aadae12"
                ),
                chunk_id="chunk_daf2232e15a1f8eb",
                sequence=0,
                content_sha256="a" * 64,
                acl_version="demo-acl-v1",
            ),
            EvidenceReference(
                tenant_id=request.tenant_id,
                source=evidence_source,
                document_id=(
                    "dsid_eab08bf08daa480cae4e2f511aadae12"
                ),
                chunk_id="chunk_c0aeb153d7689513",
                sequence=1,
                content_sha256="b" * 64,
                acl_version="demo-acl-v1",
            ),
        )

        return SearchToolResponse(
            evidence=evidence,
        )


def main() -> None:
    """运行带模拟检索的规划图，并打印各阶段动作与审计记录。"""
    # 运行上下文定义身份、预算，以及允许 Planner 使用的检索范围。
    context = AgentRuntimeContext(
        request_id="planning-graph-001",
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
            allowed_sources=frozenset(
                get_args(SourceName)
            ),
            allow_cross_source=True,
            max_top_k=5,
        ),
    )

    initial_state = create_initial_graph_state(
        goal=(
            "Investigate how project lifecycle archival "
            "should close Slack channels, snapshot "
            "Confluence pages, and archive Jira issues."
        ),
        runtime_context=context,
    )

    # Planner 使用真实模型，搜索步骤使用上面的可预测假工具。
    planner_client = OpenAICompatibleLLMClient(
        load_llm_settings()
    )
    search_tool = DemoSearchTool()

    graph = build_agent_graph(
        planner_client,
        search_tool,
    )

    # 分别保存策略校验前的模型提议和校验后的批准动作。
    model_proposal = None
    approved_action = None
    final_state = initial_state

    try:
        # values 模式会在每个节点完成后返回完整状态快照。
        for state_snapshot in graph.stream(
            initial_state,
            context=context,
            stream_mode="values",
        ):
            final_state = state_snapshot

            # 第一次出现 pending_action 时记录 Planner 的原始建议。
            if (
                model_proposal is None
                and state_snapshot["pending_action"]
                is not None
            ):
                model_proposal = dict(
                    state_snapshot["pending_action"]
                )

            # approved_action 可能在后续快照更新，因此保留最新值。
            if (
                state_snapshot["approved_action"]
                is not None
            ):
                approved_action = dict(
                    state_snapshot["approved_action"]
                )
    finally:
        planner_client.close()

    print("========== Agent执行结果 ==========")
    print("任务状态：", final_state["status"])
    print("规划步骤：", final_state["step_count"])
    print(
        "模型Token：",
        final_state["model_token_count"],
    )

    print("\n========== 模型建议 ==========")
    print(model_proposal)

    print("\n========== 系统批准 ==========")
    print(approved_action)

    print("\n========== 工具实际请求 ==========")

    # 假工具记录的请求能证明工具节点是否真的执行。
    if search_tool.requests:
        print(search_tool.requests[0])
    else:
        print("工具未被调用")

    print("\n========== 执行后的Agent状态 ==========")
    print(
        "工具调用次数：",
        final_state["tool_call_count"],
    )
    print(
        "检索切块数量：",
        final_state["retrieved_chunk_count"],
    )
    print(
        "证据引用数量：",
        len(final_state["evidence_refs"]),
    )
    print(
        "待执行动作：",
        final_state["pending_action"],
    )
    print(
        "已批准动作：",
        final_state["approved_action"],
    )

    print("\n========== 证据引用 ==========")

    # 状态只保存可审计的证据引用，不在这里伪造正文。
    for index, evidence in enumerate(
        final_state["evidence_refs"],
        start=1,
    ):
        print(
            f"E{index}："
            f"{evidence['source']} | "
            f"{evidence['document_id']} | "
            f"{evidence['chunk_id']}"
        )

    print("\n========== 工具审计记录 ==========")

    for record in final_state["tool_calls"]:
        print(record)

    print(
        "\n错误：",
        final_state["last_error_code"],
    )


if __name__ == "__main__":
    # 教程脚本被直接执行时才运行示例。
    main()
