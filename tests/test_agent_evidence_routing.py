"""验证证据审核完成后的路由规则。

这些用例说明预算、审核结论和运行状态如何共同决定回答、重试、拒答或停止执行。
"""

from __future__ import annotations

import pytest

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.evidence_routing import (
    EvidenceRoutingInvariantError,
    decide_evidence_route,
)
from enterprise_agent.agent.state import (
    EvidenceAssessmentState,
    EvidenceDecision,
    EvidenceFactState,
    create_initial_graph_state,
)


def make_context() -> AgentRuntimeContext:
    """构造使用默认预算且仅允许 Jira 的路由测试上下文。"""
    return AgentRuntimeContext(
        request_id="routing-test-001",
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


def make_assessment(
    decision: EvidenceDecision,
    judge_succeeded: bool = True,
) -> EvidenceAssessmentState:
    """按审核决策与审核是否成功构造相匹配的证据评估。"""
    if decision == "supported":
        facts = [
            EvidenceFactState(
                fact="deployment date",
                status="supported",
                conclusion="2026-08-01",
                evidence_ids=["E1"],
            )
        ]

    elif decision == "partially_supported":
        facts = [
            EvidenceFactState(
                fact="deployment date",
                status="supported",
                conclusion="2026-08-01",
                evidence_ids=["E1"],
            ),
            EvidenceFactState(
                fact="approved ticket",
                status="missing",
                conclusion=None,
                evidence_ids=[],
            ),
        ]

    elif decision == "conflicting":
        facts = [
            EvidenceFactState(
                fact="deployment date",
                status="conflicting",
                conclusion=(
                    "E1 and E2 contain "
                    "different dates."
                ),
                evidence_ids=["E1", "E2"],
            )
        ]

    else:
        facts = [
            EvidenceFactState(
                fact="deployment date",
                status="missing",
                conclusion=None,
                evidence_ids=[],
            )
        ]

    return EvidenceAssessmentState(
        decision=decision,
        facts=facts,
        reason="Test assessment.",
        judge_succeeded=judge_succeeded,
        used_model=True,
        error_code=(
            None
            if judge_succeeded
            else "MODEL_CALL_FAILED"
        ),
        model="fake-judge",
        input_tokens=100,
        output_tokens=20,
        request_id="judge-test-001",
        assessed_chunk_ids=["chunk-001"],
        assessed_at="2026-08-14T00:00:00+00:00",
    )


def make_state(
    decision: EvidenceDecision,
):
    """创建初始图状态，并注入指定决策对应的证据评估。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="What is the deployment date?",
        runtime_context=context,
    )

    state["evidence_assessment"] = (
        make_assessment(decision)
    )

    return state


def test_supported_routes_to_answer() -> None:
    """验证证据完全支持问题时路由到正式回答。"""
    state = make_state("supported")

    assert (
        decide_evidence_route(state)
        == "answer"
    )


def test_unsupported_with_budget_retries() -> None:
    """验证证据不支持但仍有预算时路由到再次搜索。"""
    state = make_state("unsupported")

    assert (
        decide_evidence_route(state)
        == "retry_search"
    )


def test_partial_with_budget_retries() -> None:
    """验证证据部分支持且仍有预算时继续补充搜索。"""
    state = make_state(
        "partially_supported"
    )

    assert (
        decide_evidence_route(state)
        == "retry_search"
    )


def test_unsupported_without_budget_abstains() -> None:
    """验证证据不支持且工具预算耗尽时路由到拒答。"""
    state = make_state("unsupported")

    state["tool_call_count"] = (
        state["budget"]["max_tool_calls"]
    )

    assert (
        decide_evidence_route(state)
        == "abstain"
    )


def test_partial_without_budget_is_partial_answer() -> None:
    """验证部分支持且步数耗尽时输出带边界的部分回答。"""
    state = make_state(
        "partially_supported"
    )

    state["step_count"] = (
        state["budget"]["max_steps"]
    )

    assert (
        decide_evidence_route(state)
        == "partial_answer"
    )


def test_conflict_routes_to_conflict_report() -> None:
    """验证证据互相冲突时路由到冲突报告。"""
    state = make_state("conflicting")

    assert (
        decide_evidence_route(state)
        == "report_conflict"
    )


def test_failed_run_stops() -> None:
    """验证已经失败的运行不再继续证据流程。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="What is the deployment date?",
        runtime_context=context,
    )
    state["status"] = "failed"

    assert (
        decide_evidence_route(state)
        == "stop"
    )


def test_judge_technical_failure_stops() -> None:
    """验证审核模型技术故障会停止，而不会被当作普通证据不足。"""
    state = make_state("unsupported")

    state["evidence_assessment"] = (
        make_assessment(
            "unsupported",
            judge_succeeded=False,
        )
    )

    assert (
        decide_evidence_route(state)
        == "stop"
    )


def test_missing_assessment_is_rejected() -> None:
    """验证缺少证据评估时抛出路由不变量错误。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="What is the deployment date?",
        runtime_context=context,
    )

    with pytest.raises(
        EvidenceRoutingInvariantError,
        match="NO_EVIDENCE_ASSESSMENT",
    ):
        decide_evidence_route(state)
