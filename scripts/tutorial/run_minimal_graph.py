"""最小 LangGraph 教程：演示运行时上下文会覆盖创建状态时的预算。"""

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RunBudget,
)
from enterprise_agent.agent.graph import (
    build_agent_graph,
)
from enterprise_agent.agent.state import (
    create_initial_graph_state,
)


def main() -> None:
    """构造最小状态并执行只包含上下文校验的 Agent 图。"""
    # principal 表示调用者身份，同一任务的前后上下文必须保持身份一致。
    principal = PrincipalContext(
        tenant_id="tenant-a",
        subject_id="alice",
        roles=frozenset({"employee"}),
        groups=frozenset(),
        authz_version="authz-v1",
    )

    # 创建初始状态时先给出一个较宽松的步骤预算。
    creation_context = AgentRuntimeContext(
        request_id="request-create-001",
        principal=principal,
        budget=RunBudget(
            max_steps=10,
        ),
    )

    # 初始状态保存目标、身份快照和创建时预算。
    initial_state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=creation_context,
    )

    # 真正运行时使用更严格的当前策略预算，图会采用该有效限制。
    current_context = AgentRuntimeContext(
        request_id="request-run-002",
        principal=principal,
        budget=RunBudget(
            max_steps=3,
        ),
    )

    # 不传外部组件时，构建可用于验证核心状态流转的最小图。
    graph = build_agent_graph()

    final_state = graph.invoke(
        initial_state,
        context=current_context,
    )

    print("执行路径：START -> validate_context -> END")
    print("任务ID：", final_state["task_id"])
    print("任务状态：", final_state["status"])
    print(
        "创建时的步骤预算：",
        initial_state["budget"]["max_steps"],
    )
    print(
        "当前企业策略预算：",
        current_context.budget.max_steps,
    )
    print(
        "图执行后的有效预算：",
        final_state["budget"]["max_steps"],
    )


if __name__ == "__main__":
    # 教程脚本被直接执行时才运行示例。
    main()
