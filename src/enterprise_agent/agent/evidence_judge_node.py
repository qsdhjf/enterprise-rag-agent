"""加载当前证据并调用审核器，判断证据对问题的支持程度。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
)
from enterprise_agent.agent.state import (
    AgentGraphState,
    EvidenceAssessmentState,
    EvidenceFactState,
    EvidenceReference,
)
from enterprise_agent.generation.evidence_judge import (
    EvidenceJudgeResult,
    judge_evidence,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
)


# 限制输入规模，避免一次审核读取过多正文或意外耗尽模型预算。
MAX_JUDGE_EVIDENCE = 10
JUDGE_TOKEN_RESERVE = 5000


class EvidenceContentResolver(Protocol):
    """
    根据证据引用临时读取证据正文。

    真实实现以后会从OpenSearch读取；
    单元测试可以使用Fake Resolver。
    """

    def resolve(
        self,
        references: Sequence[EvidenceReference],
        runtime_context: AgentRuntimeContext,
    ) -> Sequence[OpenSearchHit]:
        """按引用顺序返回正文；实现必须同时执行租户与完整性检查。"""

        ...


class EvidenceAssessmentInvariantError(
    RuntimeError
):
    """审核前发现Agent状态不一致。"""


def _now() -> str:
    """返回可持久化、带 UTC 时区的审核时间。"""

    return datetime.now(timezone.utc).isoformat()


def _serialize_result(
    result: EvidenceJudgeResult,
    references: Sequence[EvidenceReference],
) -> EvidenceAssessmentState:
    """把生成层结果复制为仅包含可持久化字段的图状态。"""

    facts = [
        EvidenceFactState(
            fact=item.fact,
            status=item.status,
            conclusion=item.conclusion,
            evidence_ids=list(item.evidence_ids),
        )
        for item in result.facts
    ]

    return EvidenceAssessmentState(
        decision=result.decision,
        facts=facts,
        reason=result.reason,
        judge_succeeded=result.judge_succeeded,
        used_model=result.used_model,
        error_code=result.error_code,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        request_id=result.request_id,
        assessed_chunk_ids=[
            reference["chunk_id"]
            for reference in references
        ],
        assessed_at=_now(),
    )


def _validate_resolved_evidence(
    references: Sequence[EvidenceReference],
    evidence: Sequence[OpenSearchHit],
) -> None:
    """确认解析出的正文与状态中的引用逐项一致且内容非空。"""

    if len(references) != len(evidence):
        raise ValueError(
            "EVIDENCE_COUNT_MISMATCH"
        )

    for reference, hit in zip(
        references,
        evidence,
        strict=True,
    ):
        if (
            hit.chunk_id != reference["chunk_id"]
            or hit.document_id
            != reference["document_id"]
            or hit.source != reference["source"]
            or hit.sequence != reference["sequence"]
        ):
            raise ValueError(
                "EVIDENCE_REFERENCE_MISMATCH"
            )

        if not hit.text.strip():
            raise ValueError(
                "EVIDENCE_TEXT_EMPTY"
            )


def _technical_failure(
    error_code: str,
    message: str,
) -> dict[str, object]:
    """构造证据审核阶段统一的技术失败状态更新。"""

    return {
        "status": "failed",
        "evidence_assessment": None,
        "last_error_code": error_code,
        "last_error_message": message,
    }


def assess_current_evidence(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
    resolver: EvidenceContentResolver,
    client: OpenAICompatibleLLMClient,
) -> dict[str, object]:
    """审核当前累计证据，并返回供 LangGraph 合并的局部状态。"""

    if state["status"] != "running":
        raise EvidenceAssessmentInvariantError(
            "RUN_NOT_ACTIVE"
        )

    principal = runtime_context.principal

    if (
        state["tenant_id"] != principal.tenant_id
        or state["subject_id"]
        != principal.subject_id
    ):
        raise EvidenceAssessmentInvariantError(
            "PRINCIPAL_MISMATCH"
        )

    if (
        state["authz_version"]
        != principal.authz_version
    ):
        raise EvidenceAssessmentInvariantError(
            "AUTHORIZATION_CONTEXT_STALE"
        )

    # 先冻结本轮引用快照，确保正文解析与最终记录对应同一批切块。
    references = tuple(state["evidence_refs"])

    if len(references) > MAX_JUDGE_EVIDENCE:
        return _technical_failure(
            "TOO_MANY_EVIDENCE_REFS",
            "Evidence judge accepts at most 10 chunks.",
        )

    # 没有引用时不需要读取正文。
    if not references:
        evidence: tuple[OpenSearchHit, ...] = ()
    else:
        # 审核前预留最坏情况下的用量；供应商不给 usage 时也能安全记账。
        remaining_tokens = (
            state["budget"]["max_model_tokens"]
            - state["model_token_count"]
        )

        if remaining_tokens < JUDGE_TOKEN_RESERVE:
            return {
                "status": "budget_exceeded",
                "evidence_assessment": None,
                "last_error_code": (
                    "INSUFFICIENT_JUDGE_TOKEN_BUDGET"
                ),
                "last_error_message": (
                    "Insufficient token budget "
                    "for evidence judging."
                ),
            }

        try:
            evidence = tuple(
                resolver.resolve(
                    references,
                    runtime_context,
                )
            )

            # Resolver 是可替换边界，不能默认信任它返回的顺序和元数据。
            _validate_resolved_evidence(
                references,
                evidence,
            )
        except Exception:
            return _technical_failure(
                "EVIDENCE_LOAD_FAILED",
                (
                    "Evidence content could not "
                    "be loaded safely."
                ),
            )

    try:
        # 这里直接复用以前完成的审核模块。
        result = judge_evidence(
            question=state["goal"],
            evidence=evidence,
            client=client,
        )
    except Exception:
        return _technical_failure(
            "EVIDENCE_JUDGE_FAILED",
            "Evidence judge failed unexpectedly.",
        )

    # 某些确定性审核路径不调用模型，因此不应虚增 token 计数。
    if not result.used_model:
        used_tokens = 0
    elif (
        result.input_tokens is None
        or result.output_tokens is None
    ):
        used_tokens = JUDGE_TOKEN_RESERVE
    else:
        used_tokens = (
            result.input_tokens
            + result.output_tokens
        )

    total_tokens = (
        state["model_token_count"]
        + used_tokens
    )

    # 即使随后发现超预算，也保留审核结果和实际用量便于审计。
    assessment = _serialize_result(
        result,
        references,
    )

    updates: dict[str, object] = {
        "evidence_assessment": assessment,
        "model_token_count": total_tokens,
    }

    if (
        total_tokens
        > state["budget"]["max_model_tokens"]
    ):
        updates.update(
            {
                "status": "budget_exceeded",
                "last_error_code": (
                    "MODEL_TOKEN_BUDGET_EXCEEDED"
                ),
                "last_error_message": (
                    "Evidence judge exceeded "
                    "the model token budget."
                ),
            }
        )

        return updates

    # 技术失败和“确实没有证据”必须区分。
    if not result.judge_succeeded:
        updates.update(
            {
                "status": "failed",
                "last_error_code": (
                    result.error_code
                    or "EVIDENCE_JUDGE_FAILED"
                ),
                "last_error_message": (
                    "Evidence judge did not "
                    "complete successfully."
                ),
            }
        )

        return updates

    updates.update(
        {
            "last_error_code": None,
            "last_error_message": None,
        }
    )

    return updates
