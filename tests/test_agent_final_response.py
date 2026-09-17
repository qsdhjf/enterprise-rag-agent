"""验证最终回答的生成、拒答和引用校验逻辑。

测试既覆盖独立响应构造器，也确认图节点会把最终结果正确写回 Agent 状态。
"""

import pytest

from enterprise_agent.agent.answer_node import (
    generate_final_response,
)
from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.final_response import (
    FinalAnswerError,
    generate_final_answer,
)
from enterprise_agent.agent.state import (
    EvidenceReference,
    create_initial_graph_state,
)
from enterprise_agent.infrastructure.llm_client import LLMResult
from enterprise_agent.retrieval.keyword_search import OpenSearchHit


def make_context() -> AgentRuntimeContext:
    """构造最终回答测试所需的固定身份、预算和检索策略。"""
    return AgentRuntimeContext(
        request_id="final-answer-test",
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


def make_hit() -> OpenSearchHit:
    """构造包含项目归档流程的一条固定检索命中。"""
    return OpenSearchHit(
        score=10.0,
        chunk_id="chunk-001",
        document_id="document-001",
        source="jira",
        title="Project lifecycle archival",
        sequence=0,
        text=(
            "Slack channels become read-only, Confluence pages are "
            "snapshotted, and Jira issues are moved to an archive."
        ),
        created_at=None,
        original_location="sources/jira/project.json",
    )


def make_state(decision: str):
    """创建图状态，并按审核决策填充证据引用及评估结果。"""
    context = make_context()
    state = create_initial_graph_state(
        goal="How should the project be archived?",
        runtime_context=context,
    )

    if decision != "unsupported":
        state["evidence_refs"] = [
            EvidenceReference(
                tenant_id="tenant-a",
                source="jira",
                document_id="document-001",
                chunk_id="chunk-001",
                sequence=0,
                content_sha256="a" * 64,
                acl_version="acl-v1",
            )
        ]

    fact_status = (
        "missing" if decision == "unsupported" else "supported"
    )
    state["evidence_assessment"] = {
        "decision": decision,
        "facts": [
            {
                "fact": "project archival process",
                "status": fact_status,
                "conclusion": (
                    None
                    if fact_status == "missing"
                    else "The process is documented."
                ),
                "evidence_ids": (
                    [] if fact_status == "missing" else ["E1"]
                ),
            }
        ],
        "reason": "test assessment",
        "judge_succeeded": True,
        "used_model": decision != "unsupported",
        "error_code": None,
        "model": "fake-judge",
        "input_tokens": 10,
        "output_tokens": 10,
        "request_id": "judge-001",
        "assessed_chunk_ids": (
            [] if decision == "unsupported" else ["chunk-001"]
        ),
        "assessed_at": "2026-08-14T00:00:00+00:00",
    }
    return context, state


class FakeResolver:
    """返回固定命中并记录解析次数的证据解析器替身。"""

    def __init__(self) -> None:
        """将解析调用计数初始化为零。"""
        self.calls = 0

    def resolve(self, references, runtime_context):
        """记录一次解析调用并返回预设的检索命中。"""
        self.calls += 1
        return (make_hit(),)


class FakeAnswerClient:
    """返回指定回答文本并记录调用次数的假回答模型。"""

    def __init__(self, answer: str) -> None:
        """保存预设回答并初始化调用计数。"""
        self.answer = answer
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """记录一次生成调用并返回带固定元数据的回答。"""
        self.calls += 1
        return LLMResult(
            text=self.answer,
            model="fake-answer",
            request_id="answer-001",
            input_tokens=100,
            output_tokens=20,
        )


class MustNotBeCalled:
    """任何属性访问都会失败，用于证明某依赖未被使用。"""

    def __getattr__(self, name):
        """报告意外的依赖属性访问并立即让测试失败。"""
        raise AssertionError(f"unexpected call: {name}")


def test_supported_answer_contains_valid_citation() -> None:
    """验证受证据支持的回答保留有效引用并调用解析器和模型。"""
    context, state = make_state("supported")
    resolver = FakeResolver()
    client = FakeAnswerClient(
        "The project uses the documented archival workflow [E1]."
    )

    result = generate_final_answer(
        state,
        context,
        resolver,
        client,
    )

    assert result.outcome == "supported"
    assert result.citations == ("E1",)
    assert result.used_model is True
    assert resolver.calls == 1
    assert client.calls == 1


def test_unsupported_answer_does_not_call_model() -> None:
    """验证证据不足时直接生成拒答且不访问模型或解析器。"""
    context, state = make_state("unsupported")

    result = generate_final_answer(
        state,
        context,
        MustNotBeCalled(),
        MustNotBeCalled(),
    )

    assert result.outcome == "unsupported"
    assert result.used_model is False
    assert result.citations == ()
    assert "证据不足" in result.answer


def test_unknown_citation_is_rejected() -> None:
    """验证回答引用不存在的证据编号时抛出引用越界错误。"""
    context, state = make_state("supported")

    with pytest.raises(
        FinalAnswerError,
        match="FINAL_ANSWER_CITATION_OUT_OF_RANGE",
    ):
        generate_final_answer(
            state,
            context,
            FakeResolver(),
            FakeAnswerClient("Unsupported citation [E9]."),
        )


def test_graph_answer_node_stores_final_response() -> None:
    """验证回答节点把最终响应及模型用量正确写回图状态。"""
    context, state = make_state("supported")

    updates = generate_final_response(
        state=state,
        runtime_context=context,
        resolver=FakeResolver(),
        client=FakeAnswerClient(
            "The archival workflow is documented [E1]."
        ),
    )

    assert updates["status"] == "completed"
    assert updates["model_token_count"] == 120

    final_response = updates["final_response"]

    assert final_response is not None
    assert final_response["outcome"] == "supported"
    assert final_response["citations"] == ["E1"]
    assert final_response["used_model"] is True
