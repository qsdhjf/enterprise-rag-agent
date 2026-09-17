"""策略门教程：展示系统如何收紧 Planner 提出的检索动作。"""

from typing import get_args
from uuid import uuid4

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
    SourceName,
)
from enterprise_agent.agent.state import (
    PlannerActionState,
    create_initial_graph_state,
)
from enterprise_agent.agent.policy import (
    evaluate_action_policy,
)


def main() -> None:
    """构造一个超出策略 top-k 的动作，并查看策略校验后的结果。"""
    # 检索策略把单次最大 top-k 设为 5。
    context = AgentRuntimeContext(
        request_id="policy-test-001",
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

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )

    # 模拟 Planner 提出 top_k=10 的宽泛搜索，故意超过策略上限。
    state["pending_action"] = PlannerActionState(
        action_id=str(uuid4()),
        kind="search_enterprise_knowledge",
        query=(
            "project lifecycle archival close Slack "
            "channels snapshot Confluence pages "
            "archive Jira issues"
        ),
        source=None,
        top_k=10,
        reason="Initial broad search.",
    )

    # 策略门不会直接信任模型输出，而是返回经过批准/收紧的状态更新。
    updates = evaluate_action_policy(
        state,
        context,
    )

    print(
        "模型建议top_k：",
        state["pending_action"]["top_k"],
    )
    print(
        "企业批准top_k：",
        updates["approved_action"]["top_k"],
    )
    print(
        "批准动作：",
        updates["approved_action"],
    )
    print(
        "错误：",
        updates.get("last_error_code"),
    )


if __name__ == "__main__":
    # 教程脚本被直接执行时才运行示例。
    main()
