"""根据证据审核结论和剩余预算选择 LangGraph 的下一条边。"""

from __future__ import annotations

from typing import Literal

from enterprise_agent.agent.state import (
    AgentGraphState,
)


# 这些字符串必须与 graph.py 中 conditional edge 的映射键保持一致。
EvidenceRoute = Literal[
    "answer",
    "retry_search",
    "partial_answer",
    "abstain",
    "report_conflict",
    "stop",
]


class EvidenceRoutingInvariantError(
    RuntimeError
):
    """证据路由时发现Agent状态不完整。"""


def _has_search_budget(
    state: AgentGraphState,
) -> bool:
    """只有所有硬预算都尚未耗尽时，才允许进入下一轮检索。"""

    budget = state["budget"]

    if state["step_count"] >= budget["max_steps"]:
        return False

    if (
        state["tool_call_count"]
        >= budget["max_tool_calls"]
    ):
        return False

    if (
        state["retrieved_chunk_count"]
        >= budget["max_retrieved_chunks"]
    ):
        return False

    if (
        state["model_token_count"]
        >= budget["max_model_tokens"]
    ):
        return False

    return True


def decide_evidence_route(
    state: AgentGraphState,
) -> EvidenceRoute:
    """把审核结论转换为回答、重试、拒答或停止路由。"""

    if state["status"] != "running":
        return "stop"

    assessment = state["evidence_assessment"]

    if assessment is None:
        raise EvidenceRoutingInvariantError(
            "NO_EVIDENCE_ASSESSMENT"
        )

    if not assessment["judge_succeeded"]:
        return "stop"

    decision = assessment["decision"]

    # 完全支持和证据冲突都无需继续搜索，可直接生成确定性下一步。
    if decision == "supported":
        return "answer"

    if decision == "conflicting":
        return "report_conflict"

    has_budget = _has_search_budget(state)

    # 部分支持/不支持时优先补证；预算耗尽后再降级回答或明确拒答。
    if decision == "partially_supported":
        if has_budget:
            return "retry_search"

        return "partial_answer"

    if decision == "unsupported":
        if has_budget:
            return "retry_search"

        return "abstain"

    raise EvidenceRoutingInvariantError(
        f"UNKNOWN_EVIDENCE_DECISION:{decision}"
    )
