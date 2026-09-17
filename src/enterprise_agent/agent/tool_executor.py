"""执行已获批准的只读搜索，并把结果安全地合并回 Agent 状态。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    SourceName,
)
from enterprise_agent.agent.state import (
    AgentGraphState,
    EvidenceReference,
    ToolCallRecord,
)


@dataclass(frozen=True, slots=True)
class SearchToolRequest:
    """传给搜索工具的可信身份快照与已审批查询参数。"""

    tenant_id: str
    subject_id: str
    authz_version: str

    query: str
    source: SourceName | None
    top_k: int
    # 该对象只由服务端执行节点注入；模型动作和 HTTP body 都不能提供它。
    principal: PrincipalContext | None = None


@dataclass(frozen=True, slots=True)
class SearchToolResponse:
    """搜索工具返回的证据引用集合；正文由 Resolver 单独管理。"""

    evidence: tuple[EvidenceReference, ...]


class AgentSearchTool(Protocol):
    """任意 Agent 搜索后端都必须实现的最小同步接口。"""

    def run(
        self,
        request: SearchToolRequest,
    ) -> SearchToolResponse:
        """执行一次只读搜索并返回证据引用。"""

        ...


class ToolExecutionInvariantError(RuntimeError):
    """执行工具前发现Agent状态不一致。"""


def _now() -> str:
    """返回工具审计记录使用的 UTC ISO 时间。"""

    return datetime.now(timezone.utc).isoformat()


def execute_approved_search(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
    tool: AgentSearchTool,
) -> dict[str, object]:
    """校验动作与预算、执行工具，并返回一次原子的状态更新。"""

    if state["status"] != "running":
        raise ToolExecutionInvariantError(
            "RUN_NOT_ACTIVE"
        )

    principal = runtime_context.principal

    if (
        state["tenant_id"] != principal.tenant_id
        or state["subject_id"] != principal.subject_id
    ):
        raise ToolExecutionInvariantError(
            "PRINCIPAL_MISMATCH"
        )

    if (
        state["authz_version"]
        != principal.authz_version
    ):
        raise ToolExecutionInvariantError(
            "AUTHORIZATION_CONTEXT_STALE"
        )

    # pending 与 approved 必须指向同一 action_id，防止审批后被替换参数。
    pending = state["pending_action"]
    approved = state["approved_action"]

    if approved is None:
        raise ToolExecutionInvariantError(
            "NO_APPROVED_ACTION"
        )

    if pending is None:
        raise ToolExecutionInvariantError(
            "NO_PENDING_ACTION"
        )

    if (
        pending["action_id"]
        != approved["action_id"]
    ):
        raise ToolExecutionInvariantError(
            "APPROVAL_ACTION_MISMATCH"
        )

    if (
        approved["kind"]
        != "search_enterprise_knowledge"
    ):
        raise ToolExecutionInvariantError(
            "APPROVED_ACTION_IS_NOT_SEARCH"
        )

    query = approved["query"]
    top_k = approved["top_k"]
    source = approved["source"]

    if (
        query is None
        or not query.strip()
        or top_k is None
    ):
        raise ToolExecutionInvariantError(
            "INVALID_APPROVED_SEARCH"
        )

    # 防止同一个 action_id 被重复执行；成功记录可安全地按幂等方式清理。
    for previous_call in state["tool_calls"]:
        if (
            previous_call["call_id"]
            == approved["action_id"]
        ):
            if previous_call["status"] == "succeeded":
                return {
                    "pending_action": None,
                    "approved_action": None,
                }

            raise ToolExecutionInvariantError(
                "ACTION_ALREADY_ATTEMPTED"
            )

    if (
        state["tool_call_count"]
        >= state["budget"]["max_tool_calls"]
    ):
        raise ToolExecutionInvariantError(
            "TOOL_BUDGET_EXHAUSTED"
        )

    remaining_chunks = (
        state["budget"]["max_retrieved_chunks"]
        - state["retrieved_chunk_count"]
    )

    if top_k > remaining_chunks:
        raise ToolExecutionInvariantError(
            "APPROVED_ACTION_EXCEEDS_CHUNK_BUDGET"
        )

    request = SearchToolRequest(
        tenant_id=state["tenant_id"],
        subject_id=state["subject_id"],
        authz_version=state["authz_version"],
        query=query.strip(),
        source=source,
        top_k=top_k,
        principal=principal,
    )

    started_at = _now()
    returned_hits = 0

    # 工具属于边界依赖，返回的数量、租户、来源与完整性字段都要复验。
    try:
        response = tool.run(request)
        evidence = list(response.evidence)
        returned_hits = len(evidence)

        if returned_hits > top_k:
            raise RuntimeError(
                "工具返回数量超过批准数量"
            )

        for item in evidence:
            if item["tenant_id"] != state["tenant_id"]:
                raise RuntimeError(
                    "工具返回了其他租户的证据"
                )

            if (
                source is not None
                and item["source"] != source
            ):
                raise RuntimeError(
                    "工具返回了未批准来源的证据"
                )

            if not item["content_sha256"]:
                raise RuntimeError(
                    "证据缺少内容哈希"
                )

            if not item["acl_version"]:
                raise RuntimeError(
                    "证据缺少ACL版本"
                )

    except Exception:
        failed_call = ToolCallRecord(
            call_id=approved["action_id"],
            step_number=state["step_count"],
            tool_name=(
                "search_enterprise_knowledge"
            ),
            status="failed",
            query=query,
            source=source,
            top_k=top_k,
            returned_hits=returned_hits,
            new_evidence_count=0,
            started_at=started_at,
            completed_at=_now(),
            error_code="TOOL_EXECUTION_FAILED",
        )

        return {
            "status": "failed",
            "pending_action": None,
            "approved_action": None,
            "tool_call_count": (
                state["tool_call_count"] + 1
            ),
            "retrieved_chunk_count": (
                state["retrieved_chunk_count"]
                + returned_hits
            ),
            "tool_calls": [failed_call],
            "last_error_code": (
                "TOOL_EXECUTION_FAILED"
            ),
            "last_error_message": (
                "Approved search tool execution failed."
            ),
        }

    # retrieved_chunk_count 统计工具实际返回量；evidence_refs 只追加去重后的新证据。
    existing_keys = {
        (
            item["tenant_id"],
            item["source"],
            item["chunk_id"],
        )
        for item in state["evidence_refs"]
    }

    new_evidence: list[EvidenceReference] = []

    for item in evidence:
        key = (
            item["tenant_id"],
            item["source"],
            item["chunk_id"],
        )

        if key not in existing_keys:
            existing_keys.add(key)
            new_evidence.append(item)

    # call_id 与 action_id 相同，便于 reducer 覆盖同一次调用的审计记录。
    succeeded_call = ToolCallRecord(
        call_id=approved["action_id"],
        step_number=state["step_count"],
        tool_name="search_enterprise_knowledge",
        status="succeeded",
        query=query,
        source=source,
        top_k=top_k,
        returned_hits=returned_hits,
        new_evidence_count=len(new_evidence),
        started_at=started_at,
        completed_at=_now(),
        error_code=None,
    )

    return {
        "pending_action": None,
        "approved_action": None,
        "tool_call_count": (
            state["tool_call_count"] + 1
        ),
        "retrieved_chunk_count": (
            state["retrieved_chunk_count"]
            + returned_hits
        ),
        "evidence_refs": new_evidence,
        "tool_calls": [succeeded_call],
        "last_error_code": None,
        "last_error_message": None,
    }
