"""在执行前审批规划动作，落实身份、来源权限和资源预算。"""

from __future__ import annotations

from typing import get_args

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    SourceName,
)
from enterprise_agent.agent.state import (
    AgentGraphState,
    PlannerActionState,
)


# 用于判断跨来源搜索是否真的覆盖了用户被授权的完整来源集合。
ALL_SOURCES = frozenset(
    get_args(SourceName)
)

def _normalize_query(
    query: str,
) -> str:
    """统一大小写和连续空白，供重复搜索检测使用。"""

    return " ".join(
        query.casefold().split()
    )


def _is_duplicate_search(
    state: AgentGraphState,
    query: str,
    source: SourceName | None,
) -> bool:
    """判断同一来源是否已经执行过语义上相同的查询文本。"""

    normalized_query = _normalize_query(
        query
    )

    for previous_call in state["tool_calls"]:
        if (
            previous_call["tool_name"]
            != "search_enterprise_knowledge"
        ):
            continue

        if previous_call["source"] != source:
            continue

        previous_query = _normalize_query(
            previous_call["query"]
        )

        if previous_query == normalized_query:
            return True

    return False

class PolicyAuthorizationError(PermissionError):
    """当前身份不能审核这个任务。"""


def _deny(
    error_code: str,
    message: str,
) -> dict[str, object]:
    """构造策略拒绝动作时的失败状态。"""

    return {
        "status": "failed",
        "approved_action": None,
        "last_error_code": error_code,
        "last_error_message": message,
    }


def _budget_exceeded(
    error_code: str,
    message: str,
) -> dict[str, object]:
    """构造资源不足时的终止状态，与普通策略拒绝区分。"""

    return {
        "status": "budget_exceeded",
        "approved_action": None,
        "last_error_code": error_code,
        "last_error_message": message,
    }


def evaluate_action_policy(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
) -> dict[str, object]:
    """在可信上下文中审批 pending_action，并返回安全的动作副本。"""

    if state["status"] != "running":
        raise ValueError(
            f"任务状态不能审核：{state['status']}"
        )

    principal = runtime_context.principal

    if (
        state["tenant_id"] != principal.tenant_id
        or state["subject_id"] != principal.subject_id
    ):
        raise PolicyAuthorizationError(
            "PRINCIPAL_MISMATCH"
        )

    if (
        state["authz_version"]
        != principal.authz_version
    ):
        raise PolicyAuthorizationError(
            "AUTHORIZATION_CONTEXT_STALE"
        )

    action = state["pending_action"]

    if action is None:
        raise ValueError("NO_PENDING_ACTION")

    # finish 也必须审批：只有审核器明确给出完整支持时才能结束调查。
    if action["kind"] == "finish":
        assessment = state["evidence_assessment"]

        if assessment is None:
            return _deny(
                "FINISH_WITHOUT_ASSESSMENT",
                (
                    "Agent cannot finish before "
                    "evidence has been assessed."
                ),
            )

        if not assessment["judge_succeeded"]:
            return _deny(
                "FINISH_AFTER_JUDGE_FAILURE",
                (
                    "Agent cannot finish after "
                    "evidence-judge failure."
                ),
            )

        if assessment["decision"] != "supported":
            return _deny(
                (
                    "FINISH_WITHOUT_"
                    "SUFFICIENT_EVIDENCE"
                ),
                (
                    "Agent cannot finish without "
                    "fully supported evidence."
                ),
            )

        approved_finish = PlannerActionState(
            action_id=action["action_id"],
            kind="finish",
            query=None,
            source=None,
            top_k=None,
            reason=action["reason"],
        )

        return {
            "approved_action": approved_finish,
            "last_error_code": None,
            "last_error_message": None,
        }

    if (
        action["kind"]
        != "search_enterprise_knowledge"
    ):
        return _deny(
            "ACTION_NOT_ALLOWED",
            "The proposed action is not allowed.",
        )

    query = action["query"]
    proposed_top_k = action["top_k"]
    source = action["source"]

    if (
        query is None
        or not query.strip()
        or proposed_top_k is None
    ):
        return _deny(
            "INVALID_SEARCH_ACTION",
            "Search action fields are invalid.",
        )

    # 阻止规划器在循环中重复消耗同一个检索动作。
    if _is_duplicate_search(
        state,
        query,
        source,
    ):
        return _deny(
            "DUPLICATE_SEARCH_ACTION",
            (
                "The same normalized query and "
                "source were already attempted."
            ),
        )
    policy = runtime_context.retrieval_policy

    if source is None:
        if not policy.allow_cross_source:
            return _deny(
                "CROSS_SOURCE_SEARCH_DENIED",
                "Cross-source search is not allowed.",
            )

        # 当前搜索工具中source=None表示搜索全部九个来源。
        # 因此只有用户拥有全部来源权限时才能批准。
        if not (ALL_SOURCES - {"uploaded_file"}) <= policy.allowed_sources:
            return _deny(
                "CROSS_SOURCE_SCOPE_UNSUPPORTED",
                (
                    "Current search executor cannot safely "
                    "restrict a partial source set."
                ),
            )

    elif source not in policy.allowed_sources:
        return _deny(
            "SOURCE_NOT_ALLOWED",
            f"Access to source '{source}' is denied.",
        )

    if (
        state["tool_call_count"]
        >= state["budget"]["max_tool_calls"]
    ):
        return _budget_exceeded(
            "MAX_TOOL_CALLS_REACHED",
            "Tool-call budget was exhausted.",
        )

    remaining_chunks = (
        state["budget"]["max_retrieved_chunks"]
        - state["retrieved_chunk_count"]
    )

    if remaining_chunks <= 0:
        return _budget_exceeded(
            "MAX_RETRIEVED_CHUNKS_REACHED",
            "Retrieved-chunk budget was exhausted.",
        )

    # 策略层可向下收紧模型提议的 top_k，但绝不会扩大它。
    approved_top_k = min(
        proposed_top_k,
        policy.max_top_k,
        remaining_chunks,
    )

    approved_search = PlannerActionState(
        action_id=action["action_id"],
        kind="search_enterprise_knowledge",
        query=query.strip(),
        source=source,
        top_k=approved_top_k,
        reason=action["reason"],
    )

    return {
        "approved_action": approved_search,
        "last_error_code": None,
        "last_error_message": None,
    }
