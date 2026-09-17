"""把最终答案生成器的结果转换为 LangGraph 可合并的状态更新。"""

from __future__ import annotations

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
)
from enterprise_agent.agent.evidence_judge_node import (
    EvidenceContentResolver,
)
from enterprise_agent.agent.final_response import (
    FinalAnswerError,
    generate_final_answer,
)
from enterprise_agent.agent.state import (
    AgentGraphState,
    FinalResponseState,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)


# 当模型供应商没有返回 token 用量时，按这个保守值记账。
ANSWER_TOKEN_RESERVE = 4000


def generate_final_response(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
    resolver: EvidenceContentResolver,
    client: OpenAICompatibleLLMClient,
) -> dict[str, object]:
    """生成最终回答，并统一处理错误、token 预算和完成状态。"""

    if state["status"] != "running":
        raise RuntimeError("RUN_NOT_ACTIVE")

    try:
        result = generate_final_answer(
            state=state,
            runtime_context=runtime_context,
            resolver=resolver,
            client=client,
        )
    except FinalAnswerError:
        return {
            "status": "failed",
            "final_response": None,
            "last_error_code": "FINAL_ANSWER_INVALID",
            "last_error_message": (
                "The final answer could not be validated."
            ),
        }
    except Exception:
        return {
            "status": "failed",
            "final_response": None,
            "last_error_code": "FINAL_ANSWER_FAILED",
            "last_error_message": (
                "The final answer could not be generated."
            ),
        }

    # “无模型回答”（拒答或冲突模板）不消耗生成模型 token。
    if not result.used_model:
        used_tokens = 0
    elif (
        result.input_tokens is None
        or result.output_tokens is None
    ):
        used_tokens = ANSWER_TOKEN_RESERVE
    else:
        used_tokens = (
            result.input_tokens
            + result.output_tokens
        )

    total_tokens = (
        state["model_token_count"]
        + used_tokens
    )

    # 先记入本次真实/估算用量，再判断是否越过整个任务的总预算。
    if (
        total_tokens
        > state["budget"]["max_model_tokens"]
    ):
        return {
            "status": "budget_exceeded",
            "final_response": None,
            "model_token_count": total_tokens,
            "last_error_code": "ANSWER_TOKEN_BUDGET_EXCEEDED",
            "last_error_message": (
                "The final answer exceeded the token budget."
            ),
        }

    # TypedDict 让最终结果与 AgentGraphState 的持久化结构保持一致。
    final_response = FinalResponseState(
        outcome=result.outcome,
        answer=result.answer,
        citations=list(result.citations),
        used_model=result.used_model,
        model=result.model,
        request_id=result.request_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )

    return {
        "status": "completed",
        "final_response": final_response,
        "model_token_count": total_tokens,
        "last_error_code": None,
        "last_error_message": None,
    }
