"""定义 LangGraph 持久化状态结构、列表合并器和初始状态工厂。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    SourceName,
)


# 以下 Literal 是状态机的有限取值集合，可让路由和类型检查保持一致。
AgentTaskStatus = Literal[
    "running",
    "awaiting_approval",
    "completed",
    "failed",
    "budget_exceeded",
    "cancelled",
]


PlannerActionKind = Literal[
    "search_enterprise_knowledge",
    "finish",
]


ToolCallStatus = Literal[
    "requested",
    "succeeded",
    "failed",
    "denied",
]


EvidenceDecision = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "conflicting",
]


EvidenceFactStatus = Literal[
    "supported",
    "missing",
    "conflicting",
]

class BudgetSnapshot(TypedDict):
    """任务创建时写入 checkpoint 的不可扩张预算快照。"""

    max_steps: int
    max_tool_calls: int
    max_runtime_seconds: int
    max_model_tokens: int
    max_retrieved_chunks: int


class PlannerActionState(TypedDict):
    """规划器动作及其审批后仍需保留的参数。"""

    action_id: str
    kind: PlannerActionKind

    query: str | None
    source: SourceName | None
    top_k: int | None

    reason: str


class EvidenceReference(TypedDict):
    """不含正文的证据定位与完整性信息。"""

    tenant_id: str
    source: SourceName

    document_id: str
    chunk_id: str
    sequence: int

    content_sha256: str
    acl_version: str


class ToolCallRecord(TypedDict):
    """一次工具调用的输入、结果计数、时间和错误审计记录。"""

    call_id: str
    step_number: int

    tool_name: str
    status: ToolCallStatus

    query: str
    source: SourceName | None
    top_k: int

    returned_hits: int
    new_evidence_count: int

    started_at: str
    completed_at: str | None

    error_code: str | None

class EvidenceFactState(TypedDict):
    """审核器对单个待验证事实给出的结论。"""

    fact: str
    status: EvidenceFactStatus
    conclusion: str | None
    evidence_ids: list[str]


class EvidenceAssessmentState(TypedDict):
    """一轮证据审核的结论、事实明细和模型调用元数据。"""

    decision: EvidenceDecision
    facts: list[EvidenceFactState]
    reason: str

    judge_succeeded: bool
    used_model: bool
    error_code: str | None

    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    request_id: str | None

    # 记录本次结论到底审核了哪些切块。
    assessed_chunk_ids: list[str]
    assessed_at: str

class FinalResponseState(TypedDict):
    """最终用户回答及其引用和生成模型元数据。"""

    outcome: EvidenceDecision
    answer: str
    citations: list[str]

    used_model: bool
    model: str | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None


def merge_evidence_references(
    existing: list[EvidenceReference],
    updates: list[EvidenceReference],
) -> list[EvidenceReference]:
    """按租户、来源和切块去重，让新引用覆盖同键旧引用。"""

    # LangGraph reducer 会在节点返回 evidence_refs 时调用此函数。
    merged = {
        (
            item["tenant_id"],
            item["source"],
            item["chunk_id"],
        ): item
        for item in existing
    }

    for item in updates:
        key = (
            item["tenant_id"],
            item["source"],
            item["chunk_id"],
        )
        merged[key] = item

    return list(merged.values())


def merge_tool_calls(
    existing: list[ToolCallRecord],
    updates: list[ToolCallRecord],
) -> list[ToolCallRecord]:
    """按 call_id 合并工具记录，同时保留首次出现的顺序。"""

    order: list[str] = []
    merged: dict[str, ToolCallRecord] = {}

    for item in [*existing, *updates]:
        call_id = item["call_id"]

        if call_id not in merged:
            order.append(call_id)

        merged[call_id] = item

    return [
        merged[call_id]
        for call_id in order
    ]


class AgentGraphState(TypedDict):
    """贯穿整张图并可写入 checkpoint 的完整 Agent 任务状态。"""

    schema_version: str

    task_id: str
    origin_request_id: str

    tenant_id: str
    subject_id: str
    authz_version: str

    goal: str
    status: AgentTaskStatus

    created_at: str
    deadline_at: str

    budget: BudgetSnapshot

    step_count: int
    tool_call_count: int
    model_token_count: int
    retrieved_chunk_count: int

    pending_action: PlannerActionState | None
    approved_action: PlannerActionState | None
    # Annotated 指定 LangGraph reducer，节点只需返回本轮新增列表。
    evidence_refs: Annotated[
        list[EvidenceReference],
        merge_evidence_references,
    ]
    evidence_assessment: EvidenceAssessmentState | None

    tool_calls: Annotated[
        list[ToolCallRecord],
        merge_tool_calls,
    ]
    final_response: FinalResponseState | None
    final_report_id: str | None

    last_error_code: str | None
    last_error_message: str | None


def create_initial_graph_state(
    goal: str,
    runtime_context: AgentRuntimeContext,
    task_id: str | None = None,
) -> AgentGraphState:
    """从可信运行上下文创建计数归零、带 UTC 截止时间的图状态。"""

    cleaned_goal = goal.strip()

    if not cleaned_goal:
        raise ValueError("Agent目标不能为空")

    now = datetime.now(timezone.utc)
    budget = runtime_context.budget
    deadline = now + timedelta(
        seconds=budget.max_runtime_seconds,
    )

    return AgentGraphState(
        schema_version="agent-graph-state-v4",
        task_id=task_id or str(uuid4()),
        origin_request_id=(
            runtime_context.request_id
        ),
        tenant_id=(
            runtime_context.principal.tenant_id
        ),
        subject_id=(
            runtime_context.principal.subject_id
        ),
        authz_version=(
            runtime_context.principal.authz_version
        ),
        goal=cleaned_goal,
        status="running",
        created_at=now.isoformat(),
        deadline_at=deadline.isoformat(),
        budget=BudgetSnapshot(
            max_steps=budget.max_steps,
            max_tool_calls=budget.max_tool_calls,
            max_runtime_seconds=(
                budget.max_runtime_seconds
            ),
            max_model_tokens=(
                budget.max_model_tokens
            ),
            max_retrieved_chunks=(
                budget.max_retrieved_chunks
            ),
        ),
        step_count=0,
        tool_call_count=0,
        model_token_count=0,
        retrieved_chunk_count=0,
        pending_action=None,
        approved_action=None,
        evidence_refs=[],
        evidence_assessment=None,
        tool_calls=[],
        final_response=None,
        final_report_id=None,
        last_error_code=None,
        last_error_message=None,
    )
