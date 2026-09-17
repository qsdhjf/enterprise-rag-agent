"""验证规划动作进入工具层之前的确定性策略检查。

测试覆盖来源权限、工具预算、重复搜索和无证据结束等安全边界。
"""

from typing import get_args

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
    SourceName,
)
from enterprise_agent.agent.state import (
    PlannerActionState,
    ToolCallRecord,
    create_initial_graph_state,
)
from enterprise_agent.agent.policy import (
    evaluate_action_policy,
)


ALL_SOURCES = frozenset(get_args(SourceName))


def make_context(
    *,
    allowed_sources: frozenset[str],
    allow_cross_source: bool = False,
    max_top_k: int = 5,
) -> AgentRuntimeContext:
    """构造用于策略测试的固定运行上下文。"""

    return AgentRuntimeContext(
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
            allowed_sources=allowed_sources,
            allow_cross_source=allow_cross_source,
            max_top_k=max_top_k,
        ),
    )


def make_search_action(
    *,
    source: str | None,
    top_k: int = 10,
) -> PlannerActionState:
    """构造待策略检查的企业知识搜索动作。"""

    return PlannerActionState(
        action_id="action-test-001",
        kind="search_enterprise_knowledge",
        query="project lifecycle archival",
        source=source,
        top_k=top_k,
        reason="需要检索相关企业证据",
    )


def test_allowed_action_is_approved_and_limited() -> None:
    """验证合法搜索会获批且返回数量受策略上限约束。"""

    context = make_context(
        allowed_sources=ALL_SOURCES,
        allow_cross_source=True,
        max_top_k=5,
    )

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )
    state["pending_action"] = make_search_action(
        source=None,
        top_k=10,
    )

    updates = evaluate_action_policy(
        state,
        context,
    )

    approved = updates["approved_action"]

    assert approved is not None
    assert approved["top_k"] == 5

    # 原始模型建议仍然保留为10，方便审计。
    assert state["pending_action"]["top_k"] == 10

    assert updates["last_error_code"] is None


def test_cross_source_search_is_denied() -> None:
    """验证策略禁用跨来源检索时会拒绝无来源搜索。"""

    context = make_context(
        allowed_sources=frozenset({"jira"}),
        allow_cross_source=False,
    )

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )
    state["pending_action"] = make_search_action(
        source=None,
    )

    updates = evaluate_action_policy(
        state,
        context,
    )

    assert updates["approved_action"] is None
    assert updates["status"] == "failed"
    assert (
        updates["last_error_code"]
        == "CROSS_SOURCE_SEARCH_DENIED"
    )


def test_unapproved_source_is_denied() -> None:
    """验证搜索未授权来源时会被策略拒绝。"""

    context = make_context(
        allowed_sources=frozenset({"jira"}),
    )

    state = create_initial_graph_state(
        goal="调查Slack归档流程",
        runtime_context=context,
    )
    state["pending_action"] = make_search_action(
        source="slack",
    )

    updates = evaluate_action_policy(
        state,
        context,
    )

    assert updates["approved_action"] is None
    assert updates["status"] == "failed"
    assert (
        updates["last_error_code"]
        == "SOURCE_NOT_ALLOWED"
    )


def test_finish_without_evidence_is_denied() -> None:
    """验证缺少评估与证据时不能直接结束调查。"""

    context = make_context(
        allowed_sources=frozenset({"jira"}),
    )

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )
    state["pending_action"] = PlannerActionState(
        action_id="action-finish-001",
        kind="finish",
        query=None,
        source=None,
        top_k=None,
        reason="调查已经完成",
    )

    updates = evaluate_action_policy(
        state,
        context,
    )

    assert updates["approved_action"] is None
    assert updates["status"] == "failed"
    assert (
            updates["last_error_code"]
            == "FINISH_WITHOUT_ASSESSMENT"
    )


def test_tool_budget_exhaustion_stops_action() -> None:
    """验证工具调用预算耗尽后不再批准搜索动作。"""

    context = make_context(
        allowed_sources=frozenset({"jira"}),
    )

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )
    state["pending_action"] = make_search_action(
        source="jira",
    )

    state["tool_call_count"] = (
        state["budget"]["max_tool_calls"]
    )

    updates = evaluate_action_policy(
        state,
        context,
    )

    assert updates["approved_action"] is None
    assert updates["status"] == "budget_exceeded"
    assert (
        updates["last_error_code"]
        == "MAX_TOOL_CALLS_REACHED"
    )

def test_duplicate_search_is_denied() -> None:
    """验证规范化后重复的搜索动作会被拒绝。"""

    context = make_context(
        allowed_sources=frozenset({"jira"}),
    )

    state = create_initial_graph_state(
        goal="Investigate project archival",
        runtime_context=context,
    )

    state["tool_calls"] = [
        ToolCallRecord(
            call_id="previous-search-001",
            step_number=1,
            tool_name=(
                "search_enterprise_knowledge"
            ),
            status="succeeded",
            query="PROJECT   lifecycle archival",
            source="jira",
            top_k=5,
            returned_hits=0,
            new_evidence_count=0,
            started_at=(
                "2026-08-14T00:00:00+00:00"
            ),
            completed_at=(
                "2026-08-14T00:00:01+00:00"
            ),
            error_code=None,
        )
    ]

    state["pending_action"] = make_search_action(
        source="jira",
        top_k=5,
    )

    updates = evaluate_action_policy(
        state,
        context,
    )

    assert updates["approved_action"] is None
    assert updates["status"] == "failed"
    assert (
        updates["last_error_code"]
        == "DUPLICATE_SEARCH_ACTION"
    )


def test_different_search_is_allowed() -> None:
    """验证查询内容不同的后续搜索仍可获得批准。"""

    context = make_context(
        allowed_sources=frozenset({"jira"}),
    )

    state = create_initial_graph_state(
        goal="Investigate project archival",
        runtime_context=context,
    )

    state["tool_calls"] = [
        ToolCallRecord(
            call_id="previous-search-001",
            step_number=1,
            tool_name=(
                "search_enterprise_knowledge"
            ),
            status="succeeded",
            query="project lifecycle archival",
            source="jira",
            top_k=5,
            returned_hits=1,
            new_evidence_count=1,
            started_at=(
                "2026-08-14T00:00:00+00:00"
            ),
            completed_at=(
                "2026-08-14T00:00:01+00:00"
            ),
            error_code=None,
        )
    ]

    new_action = make_search_action(
        source="jira",
        top_k=5,
    )
    new_action["query"] = (
        "approved change ticket deployment date"
    )

    state["pending_action"] = new_action

    updates = evaluate_action_policy(
        state,
        context,
    )

    approved = updates["approved_action"]

    assert approved is not None
    assert approved["query"] == (
        "approved change ticket deployment date"
    )
    assert updates["last_error_code"] is None
