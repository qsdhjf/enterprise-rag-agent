"""组装企业知识 Agent 的 LangGraph 节点、路由和循环。"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
)
from enterprise_agent.agent.graph_nodes import (
    validate_context_node,
)
from enterprise_agent.agent.state import (
    AgentGraphState,
)
from enterprise_agent.agent.planner import (
    plan_next_action,
)
from enterprise_agent.agent.policy import (
    evaluate_action_policy,
)
from enterprise_agent.agent.tool_executor import (
    AgentSearchTool,
    execute_approved_search,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.agent.evidence_judge_node import (
    EvidenceContentResolver,
    assess_current_evidence,
)
from enterprise_agent.agent.evidence_routing import (
    decide_evidence_route,
)
from enterprise_agent.agent.answer_node import (
    generate_final_response,
)


def _route_if_running(
    state: AgentGraphState,
) -> Literal["continue", "stop"]:
    """让失败、完成或超预算状态立即结束当前图执行。"""

    if state["status"] == "running":
        return "continue"

    return "stop"


def _route_after_policy(
    state: AgentGraphState,
) -> Literal[
    "execute_search",
    "final_answer",
    "no_progress",
    "stop",
]:
    """根据策略审批结果选择搜索、回答、无进展收尾或停止。"""

    # 重复搜索并不代表证据无效：若已有审核结果，可直接用现有证据收尾。
    if (
        state["last_error_code"]
        == "DUPLICATE_SEARCH_ACTION"
        and state["evidence_assessment"] is not None
    ):
        return "no_progress"

    if state["status"] != "running":
        return "stop"

    approved_action = state["approved_action"]

    if approved_action is None:
        return "stop"

    if (
        approved_action["kind"]
        == "search_enterprise_knowledge"
    ):
        return "execute_search"

    if approved_action["kind"] == "finish":
        return "final_answer"

    return "stop"


def build_agent_graph(
    planner_client: OpenAICompatibleLLMClient,
    search_tool: AgentSearchTool,
    evidence_resolver: EvidenceContentResolver,
    judge_client: OpenAICompatibleLLMClient,
    answer_client: OpenAICompatibleLLMClient | None = None,
):
    """构建并编译一次可注入客户端与工具实现的 Agent 状态图。"""

    def planner_node(
        state: AgentGraphState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """调用规划器提出下一步动作。"""

        return plan_next_action(
            state,
            planner_client,
            allowed_sources=tuple(sorted(runtime.context.retrieval_policy.allowed_sources)),
        )

    def policy_gate_node(
        state: AgentGraphState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """用可信运行上下文审批规划器提出的动作。"""

        return evaluate_action_policy(
            state,
            runtime.context,
        )

    def execute_search_node(
        state: AgentGraphState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """执行已经通过策略审批的只读检索。"""

        return execute_approved_search(
            state,
            runtime.context,
            search_tool,
        )

    def evidence_judge_node(
        state: AgentGraphState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """审核当前累积证据是否足以支持用户问题。"""

        return assess_current_evidence(
            state,
            runtime.context,
            evidence_resolver,
            judge_client,
        )

    def final_answer_node(
        state: AgentGraphState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """生成最终回答；未配置回答模型时仅结束图（便于分阶段运行）。"""

        if answer_client is None:
            return {
                "status": "completed",
            }

        return generate_final_response(
            state=state,
            runtime_context=runtime.context,
            resolver=evidence_resolver,
            client=answer_client,
        )

    def no_progress_node(
        state: AgentGraphState,
    ) -> dict[str, object]:
        """清除重复动作，使已有证据可以进入最终回答节点。"""

        return {
            "status": "running",
            "pending_action": None,
            "approved_action": None,
        }

    # Runtime context 不写入 checkpoint；图状态只保存恢复执行所需的快照。
    builder = StateGraph(
        AgentGraphState,
        context_schema=AgentRuntimeContext,
    )

    builder.add_node(
        "validate_context",
        validate_context_node,
    )
    builder.add_node(
        "planner",
        planner_node,
    )
    builder.add_node(
        "policy_gate",
        policy_gate_node,
    )
    builder.add_node(
        "execute_search",
        execute_search_node,
    )
    builder.add_node(
        "final_answer",
        final_answer_node,
    )
    builder.add_node(
        "no_progress",
        no_progress_node,
    )
    # 每一轮都先校验身份、授权版本、截止时间和累计预算。
    builder.add_edge(
        START,
        "validate_context",
    )

    builder.add_conditional_edges(
        "validate_context",
        _route_if_running,
        {
            "continue": "planner",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "planner",
        _route_if_running,
        {
            "continue": "policy_gate",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "policy_gate",
        _route_after_policy,
        {
            "execute_search": "execute_search",
            "final_answer": "final_answer",
            "no_progress": "no_progress",
            "stop": END,
        },
    )
    # 搜索后必须经过证据审核；证据不足且仍有预算时循环回入口。
    builder.add_node(
        "evidence_judge",
        evidence_judge_node,
    )
    builder.add_edge(
        "execute_search",
        "evidence_judge",
    )
    builder.add_conditional_edges(
        "evidence_judge",
        decide_evidence_route,
        {
            "answer": "final_answer",
            "retry_search": "validate_context",
            "partial_answer": "final_answer",
            "abstain": "final_answer",
            "report_conflict": "final_answer",
            "stop": END,
        },
    )

    builder.add_edge(
        "no_progress",
        "final_answer",
    )
    builder.add_edge(
        "final_answer",
        END,
    )
    return builder.compile(name="enterprise-agent")
