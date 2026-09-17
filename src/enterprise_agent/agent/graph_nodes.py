"""提供图入口处的上下文、身份、时间和预算一致性校验。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langgraph.runtime import Runtime

from enterprise_agent.agent.contracts import AgentRuntimeContext
from enterprise_agent.agent.state import (
    AgentGraphState,
    BudgetSnapshot,
)


SUPPORTED_STATE_SCHEMA_VERSION = "agent-graph-state-v4"


class AgentContextError(RuntimeError):
    """Checkpoint 状态无效或不能继续执行。"""


class AgentAuthorizationError(PermissionError):
    """当前用户无权运行或恢复这个 Agent 任务。"""


def _parse_utc_timestamp(
    field_name: str,
    value: str,
) -> datetime:
    """解析 checkpoint 时间戳，并统一转换为 UTC。"""

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AgentContextError(
            f"INVALID_TIMESTAMP:{field_name}"
        ) from exc

    if timestamp.tzinfo is None:
        raise AgentContextError(
            f"TIMEZONE_REQUIRED:{field_name}"
        )

    return timestamp.astimezone(timezone.utc)


def _get_effective_budget(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
) -> BudgetSnapshot:
    """逐项取持久化预算与当前服务端预算的较小值。"""

    stored = state["budget"]
    current = runtime_context.budget

    return BudgetSnapshot(
        max_steps=min(
            stored["max_steps"],
            current.max_steps,
        ),
        max_tool_calls=min(
            stored["max_tool_calls"],
            current.max_tool_calls,
        ),
        max_runtime_seconds=min(
            stored["max_runtime_seconds"],
            current.max_runtime_seconds,
        ),
        max_model_tokens=min(
            stored["max_model_tokens"],
            current.max_model_tokens,
        ),
        max_retrieved_chunks=min(
            stored["max_retrieved_chunks"],
            current.max_retrieved_chunks,
        ),
    )


def validate_context(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
) -> dict[str, object]:
    """验证恢复状态仍属于当前用户，且未越过任何运行上限。"""

    if (
        state["schema_version"]
        != SUPPORTED_STATE_SCHEMA_VERSION
    ):
        raise AgentContextError(
            "STATE_SCHEMA_UNSUPPORTED"
        )

    if state["status"] != "running":
        raise AgentContextError(
            f"RUN_NOT_ACTIVE:{state['status']}"
        )

    principal = runtime_context.principal

    if (
        state["tenant_id"] != principal.tenant_id
        or state["subject_id"] != principal.subject_id
    ):
        raise AgentAuthorizationError(
            "PRINCIPAL_MISMATCH"
        )

    if (
        state["authz_version"]
        != principal.authz_version
    ):
        raise AgentAuthorizationError(
            "AUTHORIZATION_CONTEXT_STALE"
        )

    # 恢复旧任务时只允许收紧限制，不能借新上下文放大原始预算。
    effective_budget = _get_effective_budget(
        state,
        runtime_context,
    )

    created_at = _parse_utc_timestamp(
        "created_at",
        state["created_at"],
    )
    stored_deadline = _parse_utc_timestamp(
        "deadline_at",
        state["deadline_at"],
    )

    # 同理，实际截止时间取 checkpoint 截止时间和当前策略截止时间的较早者。
    policy_deadline = created_at + timedelta(
        seconds=effective_budget[
            "max_runtime_seconds"
        ],
    )
    effective_deadline = min(
        stored_deadline,
        policy_deadline,
    )

    if datetime.now(timezone.utc) >= effective_deadline:
        return {
            "budget": effective_budget,
            "deadline_at": effective_deadline.isoformat(),
            "status": "budget_exceeded",
            "last_error_code": "DEADLINE_EXCEEDED",
            "last_error_message": (
                "The Agent task exceeded its deadline."
            ),
        }

    # 用同一张表检查所有累计计数，便于新增预算维度时保持一致。
    usage_limits = [
        (
            "step_count",
            state["step_count"],
            effective_budget["max_steps"],
        ),
        (
            "tool_call_count",
            state["tool_call_count"],
            effective_budget["max_tool_calls"],
        ),
        (
            "model_token_count",
            state["model_token_count"],
            effective_budget["max_model_tokens"],
        ),
        (
            "retrieved_chunk_count",
            state["retrieved_chunk_count"],
            effective_budget["max_retrieved_chunks"],
        ),
    ]

    for counter_name, used, limit in usage_limits:
        if used < 0:
            raise AgentContextError(
                f"NEGATIVE_COUNTER:{counter_name}"
            )

        if used > limit:
            return {
                "budget": effective_budget,
                "deadline_at": (
                    effective_deadline.isoformat()
                ),
                "status": "budget_exceeded",
                "last_error_code": (
                    f"{counter_name.upper()}_EXCEEDED"
                ),
                "last_error_message": (
                    "Persisted usage exceeds "
                    "the effective run budget."
                ),
            }

    return {
        "budget": effective_budget,
        "deadline_at": effective_deadline.isoformat(),
    }


def validate_context_node(
    state: AgentGraphState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, object]:
    """把 LangGraph 的 Runtime 包装解开后复用纯校验函数。"""

    return validate_context(
        state,
        runtime.context,
    )
