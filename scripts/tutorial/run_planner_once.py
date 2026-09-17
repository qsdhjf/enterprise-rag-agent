"""Planner 单步教程：调用一次模型并查看它建议的下一步动作。"""

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RunBudget,
)
from enterprise_agent.agent.state import (
    create_initial_graph_state,
)
from enterprise_agent.agent.planner import (
    plan_next_action,
)
from enterprise_agent.infrastructure.config import load_llm_settings
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)


def main() -> None:
    """创建演示状态，运行一次 Planner，并打印状态增量。"""
    # 上下文携带请求身份和执行预算，Planner 不能绕过这些限制。
    context = AgentRuntimeContext(
        request_id="planner-test-001",
        principal=PrincipalContext(
            tenant_id="tenant-a",
            subject_id="alice",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="authz-v1",
        ),
        budget=RunBudget(),
    )

    # goal 描述待调查任务；这里故意涉及多个企业数据源。
    state = create_initial_graph_state(
        goal=(
            "Investigate how project lifecycle archival "
            "should close Slack channels, snapshot "
            "Confluence pages, and archive Jira issues."
        ),
        runtime_context=context,
    )

    # Planner 通过统一的 OpenAI 兼容客户端请求配置中的模型。
    client = OpenAICompatibleLLMClient(
        load_llm_settings()
    )

    try:
        # 单步函数只返回需要合并进图状态的字段，不直接执行工具。
        updates = plan_next_action(
            state,
            client,
        )
    finally:
        # 模型调用成功或失败都要关闭 HTTP 客户端。
        client.close()

    print("Status:", updates.get("status", "running"))
    print("Step count:", updates.get("step_count"))
    print(
        "Model tokens:",
        updates.get("model_token_count"),
    )
    print(
        "Pending action:",
        updates.get("pending_action"),
    )
    print(
        "Error:",
        updates.get("last_error_code"),
    )


if __name__ == "__main__":
    # 教程脚本被直接执行时才运行示例。
    main()
