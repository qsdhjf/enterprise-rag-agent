"""验证图中前置守卫节点对身份、截止时间和预算的处理。

这些测试用于确保请求在进入昂贵的规划与检索步骤前已经满足运行约束。
"""

from datetime import datetime, timedelta, timezone

import pytest

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RunBudget,
)
from enterprise_agent.agent.graph_nodes import (
    AgentAuthorizationError,
    validate_context,
)
from enterprise_agent.agent.state import (
    create_initial_graph_state,
)


def make_runtime_context(
    *,
    tenant_id: str = "tenant-a",
    subject_id: str = "alice",
    authz_version: str = "authz-v1",
    max_steps: int = 5,
) -> AgentRuntimeContext:
    """按可覆盖的身份与步数参数构造测试运行上下文。"""
    return AgentRuntimeContext(
        request_id="request-test-001",
        principal=PrincipalContext(
            tenant_id=tenant_id,
            subject_id=subject_id,
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version=authz_version,
        ),
        budget=RunBudget(
            max_steps=max_steps,
        ),
    )


def test_original_user_can_continue() -> None:
    """验证原始用户身份一致时上下文校验允许图继续执行。"""
    context = make_runtime_context()

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )

    updates = validate_context(
        state,
        context,
    )

    assert "status" not in updates
    assert updates["budget"]["max_steps"] == 5


def test_different_user_is_rejected() -> None:
    """验证后续请求更换用户时抛出主体不匹配授权错误。"""
    original_context = make_runtime_context(
        subject_id="alice",
    )

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=original_context,
    )

    attacker_context = make_runtime_context(
        subject_id="bob",
    )

    with pytest.raises(
        AgentAuthorizationError,
        match="PRINCIPAL_MISMATCH",
    ):
        validate_context(
            state,
            attacker_context,
        )


def test_expired_task_is_stopped() -> None:
    """验证已超过截止时间的任务被标记为预算超限并停止。"""
    context = make_runtime_context()

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )

    state["deadline_at"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=1)
    ).isoformat()

    updates = validate_context(
        state,
        context,
    )

    assert updates["status"] == "budget_exceeded"
    assert (
        updates["last_error_code"]
        == "DEADLINE_EXCEEDED"
    )


def test_new_policy_can_tighten_budget() -> None:
    """验证新的运行策略可以收紧但不会放宽原有步数预算。"""
    original_context = make_runtime_context(
        max_steps=10,
    )

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=original_context,
    )

    stricter_context = make_runtime_context(
        max_steps=3,
    )

    updates = validate_context(
        state,
        stricter_context,
    )

    assert updates["budget"]["max_steps"] == 3
