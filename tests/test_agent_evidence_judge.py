"""验证证据审核节点如何把检索结果转换成可回答性结论。

测试用伪客户端隔离真实模型，重点覆盖空证据、模型故障、证据错配和正常支持四类边界。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.evidence_judge_node import (
    EvidenceContentResolver,
    assess_current_evidence,
)
from enterprise_agent.agent.state import (
    EvidenceReference,
    create_initial_graph_state,
)
from enterprise_agent.infrastructure.llm_client import (
    LLMResult,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
)


class FakeEvidenceResolver:
    """返回预设检索正文并统计解析次数的证据解析器替身。"""

    def __init__(
        self,
        evidence: Sequence[OpenSearchHit],
    ) -> None:
        """冻结预设证据序列，并初始化解析调用计数。"""
        self.evidence = tuple(evidence)
        self.calls = 0

    def resolve(
        self,
        references: Sequence[EvidenceReference],
        runtime_context: AgentRuntimeContext,
    ) -> Sequence[OpenSearchHit]:
        """记录一次解析请求并返回预设证据正文。"""
        self.calls += 1
        return self.evidence


class FakeJudgeClient:
    """始终返回 supported 审核结论的假模型客户端。"""

    def __init__(self) -> None:
        """将审核模型调用计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """返回支持全部事实的合法审核结果及固定用量。"""
        self.calls += 1

        return LLMResult(
            text=json.dumps(
                {
                    "decision": "supported",
                    "facts": [
                        {
                            "fact": "project archival process",
                            "status": "supported",
                            "conclusion": (
                                "The evidence describes "
                                "the archival process."
                            ),
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "reason": (
                        "All requested facts are supported."
                    ),
                }
            ),
            model="fake-judge",
            request_id="judge-request-001",
            input_tokens=200,
            output_tokens=50,
        )


class BrokenJudgeClient:
    """用于模拟审核模型调用发生运行时故障的客户端。"""

    def __init__(self) -> None:
        """将失败调用计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs):
        """记录调用后抛出模拟的审核模型异常。"""
        self.calls += 1
        raise RuntimeError(
            "Simulated judge model failure"
        )


class NeverCalledClient:
    """用于断言无证据或证据错配时模型不会被调用。"""

    def __init__(self) -> None:
        """将意外调用计数初始化为零。"""
        self.calls = 0

    def generate(self, **kwargs):
        """记录意外调用并立即让测试失败。"""
        self.calls += 1
        raise AssertionError(
            "Model must not be called"
        )


def make_context() -> AgentRuntimeContext:
    """构造证据审核节点测试使用的固定运行上下文。"""
    return AgentRuntimeContext(
        request_id="judge-node-test-001",
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


def make_reference(
    chunk_id: str = "chunk-001",
) -> EvidenceReference:
    """构造当前租户下、块编号可覆盖的证据引用。"""
    return EvidenceReference(
        tenant_id="tenant-a",
        source="jira",
        document_id="document-001",
        chunk_id=chunk_id,
        sequence=0,
        content_sha256="a" * 64,
        acl_version="acl-v1",
    )


def make_hit(
    chunk_id: str = "chunk-001",
) -> OpenSearchHit:
    """构造与指定块编号对应的项目归档检索正文。"""
    return OpenSearchHit(
        score=1.0,
        chunk_id=chunk_id,
        document_id="document-001",
        source="jira",
        title="Project archival process",
        sequence=0,
        text=(
            "The project archival design requires "
            "closing Slack channels and archiving "
            "related Jira issues."
        ),
        created_at=None,
        original_location=None,
    )


def test_empty_evidence_does_not_call_model() -> None:
    """验证空证据直接形成正常 unsupported 结论且不调用依赖。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="How should project archival work?",
        runtime_context=context,
    )

    resolver = FakeEvidenceResolver(
        evidence=(),
    )
    client = NeverCalledClient()

    updates = assess_current_evidence(
        state,
        context,
        resolver,
        client,
    )

    assessment = updates["evidence_assessment"]

    assert assessment is not None
    assert assessment["decision"] == "unsupported"
    assert assessment["judge_succeeded"] is True
    assert assessment["used_model"] is False
    assert assessment["assessed_chunk_ids"] == []

    assert updates["model_token_count"] == 0
    assert resolver.calls == 0
    assert client.calls == 0


def test_supported_evidence_is_saved() -> None:
    """验证受支持的审核结论、事实、块编号和 Token 均写回状态。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="How should project archival work?",
        runtime_context=context,
    )
    state["evidence_refs"] = [
        make_reference()
    ]

    resolver = FakeEvidenceResolver(
        evidence=(make_hit(),),
    )
    client = FakeJudgeClient()

    updates = assess_current_evidence(
        state,
        context,
        resolver,
        client,
    )

    assessment = updates["evidence_assessment"]

    assert resolver.calls == 1
    assert client.calls == 1

    assert assessment is not None
    assert assessment["decision"] == "supported"
    assert assessment["judge_succeeded"] is True
    assert assessment["used_model"] is True
    assert (
        assessment["assessed_chunk_ids"]
        == ["chunk-001"]
    )

    assert len(assessment["facts"]) == 1
    assert (
        assessment["facts"][0]["status"]
        == "supported"
    )
    assert (
        assessment["facts"][0]["evidence_ids"]
        == ["E1"]
    )

    # 输入200 + 输出50。
    assert updates["model_token_count"] == 250
    assert updates["last_error_code"] is None


def test_model_failure_is_not_normal_unsupported() -> None:
    """验证模型故障被记录为技术失败，而非普通的证据不足。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="How should project archival work?",
        runtime_context=context,
    )
    state["evidence_refs"] = [
        make_reference()
    ]

    resolver = FakeEvidenceResolver(
        evidence=(make_hit(),),
    )
    client = BrokenJudgeClient()

    updates = assess_current_evidence(
        state,
        context,
        resolver,
        client,
    )

    assessment = updates["evidence_assessment"]

    assert assessment is not None
    assert assessment["decision"] == "unsupported"
    assert assessment["judge_succeeded"] is False
    assert (
        assessment["error_code"]
        == "MODEL_CALL_FAILED"
    )

    assert updates["status"] == "failed"
    assert (
        updates["last_error_code"]
        == "MODEL_CALL_FAILED"
    )

    # 模型调用失败后无法获得准确Token，
    # 因此按预留量5000计费，防止绕过预算。
    assert updates["model_token_count"] == 5000


def test_mismatched_evidence_is_rejected() -> None:
    """验证解析正文与引用块不匹配时失败且不交给审核模型。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="How should project archival work?",
        runtime_context=context,
    )
    state["evidence_refs"] = [
        make_reference(
            chunk_id="expected-chunk"
        )
    ]

    resolver = FakeEvidenceResolver(
        evidence=(
            make_hit(
                chunk_id="wrong-chunk"
            ),
        )
    )
    client = NeverCalledClient()

    updates = assess_current_evidence(
        state,
        context,
        resolver,
        client,
    )

    assert updates["status"] == "failed"
    assert (
        updates["last_error_code"]
        == "EVIDENCE_LOAD_FAILED"
    )

    # 引用和正文不匹配时，绝不能交给审核模型。
    assert resolver.calls == 1
    assert client.calls == 0
