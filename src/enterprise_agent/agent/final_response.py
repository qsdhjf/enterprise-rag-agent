"""依据审核后的证据生成可引用、可校验的最终回答。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
)
from enterprise_agent.agent.evidence_judge_node import (
    EvidenceContentResolver,
)
from enterprise_agent.agent.state import (
    AgentGraphState,
    EvidenceDecision,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.generation.prompt_builder import build_rag_prompt


@dataclass(frozen=True, slots=True)
class FinalAnswerResult:
    """回答文本及其证据引用、模型来源和 token 用量。"""

    outcome: EvidenceDecision
    answer: str
    citations: tuple[str, ...]

    used_model: bool
    model: str | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None


class FinalAnswerError(RuntimeError):
    """最终答案不能被安全生成。"""


def _extract_citations(
    answer: str,
    evidence_count: int,
) -> tuple[str, ...]:
    """提取 ``[E1]`` 格式的引用，并拒绝缺失或越界编号。"""

    numbers = re.findall(r"\[E([1-9]\d*)\]", answer)

    if not numbers:
        raise FinalAnswerError(
            "FINAL_ANSWER_HAS_NO_CITATIONS"
        )

    # 保留回答中的首次出现顺序，同时去除重复引用。
    citations: list[str] = []

    for number_text in numbers:
        number = int(number_text)

        if number > evidence_count:
            raise FinalAnswerError(
                "FINAL_ANSWER_CITATION_OUT_OF_RANGE"
            )

        citation = f"E{number}"

        if citation not in citations:
            citations.append(citation)

    return tuple(citations)


def _build_unsupported_answer(
    state: AgentGraphState,
) -> FinalAnswerResult:
    """为证据不足场景生成不调用模型的明确拒答。"""

    assessment = state["evidence_assessment"]

    if assessment is None:
        raise FinalAnswerError("NO_EVIDENCE_ASSESSMENT")

    missing_facts = [
        fact["fact"]
        for fact in assessment["facts"]
        if fact["status"] == "missing"
    ]

    answer = "现有企业知识库证据不足，无法可靠回答这个问题。"

    if missing_facts:
        answer += "\n缺失信息：\n- " + "\n- ".join(
            missing_facts
        )

    return FinalAnswerResult(
        outcome="unsupported",
        answer=answer,
        citations=(),
        used_model=False,
        model=None,
        request_id=None,
        input_tokens=None,
        output_tokens=None,
    )


def _build_conflict_answer(
    state: AgentGraphState,
) -> FinalAnswerResult:
    """把冲突事实及其证据编号渲染为确定性回答。"""

    assessment = state["evidence_assessment"]

    if assessment is None:
        raise FinalAnswerError("NO_EVIDENCE_ASSESSMENT")

    lines = [
        "检索到的企业证据存在冲突，暂时不能给出唯一结论。"
    ]
    citations: list[str] = []

    for fact in assessment["facts"]:
        if fact["status"] != "conflicting":
            continue

        rendered_ids = [
            f"[{evidence_id}]"
            for evidence_id in fact["evidence_ids"]
        ]
        lines.append(
            f"- {fact['fact']}：证据存在冲突 "
            + " ".join(rendered_ids)
        )

        for evidence_id in fact["evidence_ids"]:
            if evidence_id not in citations:
                citations.append(evidence_id)

    return FinalAnswerResult(
        outcome="conflicting",
        answer="\n".join(lines),
        citations=tuple(citations),
        used_model=False,
        model=None,
        request_id=None,
        input_tokens=None,
        output_tokens=None,
    )


def generate_final_answer(
    state: AgentGraphState,
    runtime_context: AgentRuntimeContext,
    resolver: EvidenceContentResolver,
    client: OpenAICompatibleLLMClient,
    max_output_tokens: int = 800,
) -> FinalAnswerResult:
    """按审核结论生成拒答、冲突报告或带引用的模型回答。"""

    assessment = state["evidence_assessment"]

    if assessment is None:
        raise FinalAnswerError("NO_EVIDENCE_ASSESSMENT")

    if not assessment["judge_succeeded"]:
        raise FinalAnswerError(
            "EVIDENCE_JUDGE_NOT_SUCCESSFUL"
        )

    decision = assessment["decision"]

    # 无证据和冲突场景使用模板，避免模型猜测出一个看似确定的结论。
    if decision == "unsupported":
        return _build_unsupported_answer(state)

    if decision == "conflicting":
        return _build_conflict_answer(state)

    references = tuple(state["evidence_refs"])

    if not references:
        raise FinalAnswerError(
            "FINAL_ANSWER_HAS_NO_EVIDENCE"
        )

    # 状态只保存证据引用，真正回答前才通过受控 Resolver 读取正文。
    try:
        evidence = tuple(
            resolver.resolve(
                references,
                runtime_context,
            )
        )
    except Exception as error:
        raise FinalAnswerError(
            "FINAL_EVIDENCE_LOAD_FAILED"
        ) from error

    if len(evidence) != len(references):
        raise FinalAnswerError(
            "FINAL_EVIDENCE_COUNT_MISMATCH"
        )

    prompt = build_rag_prompt(
        question=state["goal"],
        results=evidence,
    )

    # 审核结论作为额外约束，告诉回答模型可以回答到什么边界。
    if decision == "partially_supported":
        extra_instruction = (
            "\n\n证据审核结果为部分支持。只回答有证据支持的部分，"
            "并明确列出仍然缺失的信息，不得补全或猜测。"
        )
    else:
        extra_instruction = (
            "\n\n证据审核结果为完整支持。直接回答用户问题，"
            "每个关键结论必须使用 [E1] 形式标注证据。"
        )

    result = client.generate(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt + extra_instruction,
        max_output_tokens=max_output_tokens,
    )

    # 模型输出在进入最终状态前必须通过引用格式和范围校验。
    citations = _extract_citations(
        result.text,
        evidence_count=len(evidence),
    )

    return FinalAnswerResult(
        outcome=decision,
        answer=result.text,
        citations=citations,
        used_model=True,
        model=result.model,
        request_id=result.request_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
