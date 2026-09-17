"""验证工具执行器只运行已获策略批准的搜索动作。

用例还检查幂等执行、租户隔离和检索证据写回状态等关键约束。
"""

from __future__ import annotations

import pytest

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.state import (
    EvidenceReference,
    PlannerActionState,
    create_initial_graph_state,
)
from enterprise_agent.agent.tool_executor import (
    SearchToolRequest,
    SearchToolResponse,
    ToolExecutionInvariantError,
    execute_approved_search,
)


class FakeSearchTool:
    """返回预设证据并保存所有请求的搜索工具替身。"""

    def __init__(
        self,
        evidence: tuple[EvidenceReference, ...],
    ) -> None:
        """保存预设证据，并创建空的请求记录列表。"""
        self.evidence = evidence
        self.requests: list[SearchToolRequest] = []

    def run(
        self,
        request: SearchToolRequest,
    ) -> SearchToolResponse:
        """记录获批搜索请求并用预设证据构造响应。"""
        self.requests.append(request)

        return SearchToolResponse(
            evidence=self.evidence,
        )


def make_context() -> AgentRuntimeContext:
    """构造工具执行测试使用的租户身份和 Jira 检索策略。"""
    return AgentRuntimeContext(
        request_id="executor-test-001",
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


def make_ready_state():
    """创建同时包含模型提案和策略批准动作的可执行状态。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="调查项目归档流程",
        runtime_context=context,
    )

    state["pending_action"] = PlannerActionState(
        action_id="action-search-001",
        kind="search_enterprise_knowledge",
        query="project lifecycle archival",
        source="jira",
        top_k=10,
        reason="需要检索企业证据",
    )

    state["approved_action"] = PlannerActionState(
        action_id="action-search-001",
        kind="search_enterprise_knowledge",
        query="project lifecycle archival",
        source="jira",
        top_k=5,
        reason="需要检索企业证据",
    )

    return state, context


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


def test_executor_uses_approved_action() -> None:
    """验证执行器采用策略收紧后的动作并写回证据与调用记录。"""
    state, context = make_ready_state()
    tool = FakeSearchTool(make_evidence())

    updates = execute_approved_search(
        state,
        context,
        tool,
    )

    assert len(tool.requests) == 1

    request = tool.requests[0]

    # 模型建议10条，但Executor必须执行批准后的5条。
    assert request.top_k == 5
    assert request.source == "jira"
    assert request.tenant_id == "tenant-a"
    assert request.subject_id == "alice"

    assert updates["tool_call_count"] == 1
    assert updates["retrieved_chunk_count"] == 2
    assert len(updates["evidence_refs"]) == 2

    assert updates["pending_action"] is None
    assert updates["approved_action"] is None

    call_record = updates["tool_calls"][0]

    assert call_record["status"] == "succeeded"
    assert call_record["returned_hits"] == 2
    assert call_record["new_evidence_count"] == 2


def test_executor_rejects_unapproved_action() -> None:
    """验证没有批准动作时拒绝执行且真实搜索工具不被调用。"""
    state, context = make_ready_state()
    state["approved_action"] = None

    tool = FakeSearchTool(make_evidence())

    with pytest.raises(
        ToolExecutionInvariantError,
        match="NO_APPROVED_ACTION",
    ):
        execute_approved_search(
            state,
            context,
            tool,
        )

    # 没有批准动作，真实工具绝不能被调用。
    assert len(tool.requests) == 0


def test_succeeded_action_is_not_executed_twice() -> None:
    """验证已成功动作即使因消息重放再次到达也不会重复执行。"""
    state, context = make_ready_state()
    tool = FakeSearchTool(make_evidence())

    original_pending = state["pending_action"]
    original_approved = state["approved_action"]

    first_updates = execute_approved_search(
        state,
        context,
        tool,
    )

    # 模拟LangGraph把第一次执行结果合并进状态。
    state.update(first_updates)

    # 模拟同一个action_id因为消息重放再次到达。
    state["pending_action"] = original_pending
    state["approved_action"] = original_approved

    second_updates = execute_approved_search(
        state,
        context,
        tool,
    )

    # 工具仍然只调用过一次。
    assert len(tool.requests) == 1

    assert second_updates == {
        "pending_action": None,
        "approved_action": None,
    }


def test_foreign_tenant_evidence_fails_closed() -> None:
    """验证搜索返回其他租户的证据时执行器保守地标记失败。"""
    state, context = make_ready_state()

    foreign_evidence = (
        EvidenceReference(
            tenant_id="tenant-b",
            source="jira",
            document_id="secret-document",
            chunk_id="secret-chunk",
            sequence=0,
            content_sha256="c" * 64,
            acl_version="acl-v1",
        ),
    )

    tool = FakeSearchTool(foreign_evidence)

    updates = execute_approved_search(
        state,
        context,
        tool,
    )

    assert len(tool.requests) == 1
    assert updates["status"] == "failed"
    assert updates["tool_call_count"] == 1
    assert updates["approved_action"] is None
    assert (
        updates["last_error_code"]
        == "TOOL_EXECUTION_FAILED"
    )
