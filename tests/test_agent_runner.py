"""验证 AgentRunner 对范围分类、图执行和错误结果的统一编排。

外部模型与搜索依赖均由伪对象替代，因此测试只观察运行器的调用顺序和失败关闭行为。
"""

import json

import pytest

import enterprise_agent.agent.runner as runner_module
from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.state import (
    EvidenceReference,
)
from enterprise_agent.agent.runner import (
    AgentRunner,
    AgentRunnerUnavailable,
)
from enterprise_agent.agent.tool_executor import (
    SearchToolResponse,
)
from enterprise_agent.infrastructure.llm_client import LLMResult
from enterprise_agent.retrieval.keyword_search import OpenSearchHit


def make_context() -> AgentRuntimeContext:
    """构造运行器测试共用的权限、预算与检索策略。"""

    return AgentRuntimeContext(
        request_id="api-request-001",
        principal=PrincipalContext(
            tenant_id="tenant-a",
            subject_id="alice",
            roles=frozenset({"employee"}),
            groups=frozenset(),
            authz_version="authz-v1",
        ),
        budget=RunBudget(
            max_steps=2,
            max_tool_calls=2,
            max_runtime_seconds=120,
            max_model_tokens=12000,
            max_retrieved_chunks=10,
        ),
        retrieval_policy=RetrievalPolicy(
            policy_version="policy-v1",
            allowed_sources=frozenset({"jira"}),
            allow_cross_source=False,
            max_top_k=5,
        ),
    )


class MustNotBeCalled:
    """提供一旦调用模型就让测试失败的哨兵客户端。"""

    def generate(self, **kwargs):
        """在意外触发模型调用时立即报告测试失败。"""

        raise AssertionError("LLM must not be called")


class ScopeClient:
    """模拟仅返回指定范围分类的模型客户端。"""

    def __init__(self, category: str) -> None:
        """保存待返回的范围类别并初始化调用计数。"""

        self.category = category
        self.calls = 0

    def generate(self, **kwargs) -> LLMResult:
        """返回固定范围分类并累计调用次数。"""

        self.calls += 1
        return LLMResult(
            text=json.dumps(
                {
                    "category": self.category,
                    "reason": "test classification",
                }
            ),
            model="fake-scope",
            request_id="scope-001",
            input_tokens=10,
            output_tokens=5,
        )


class BrokenScopeClient:
    """模拟范围分类服务不可用的模型客户端。"""

    def generate(self, **kwargs):
        """抛出异常以模拟范围分类服务故障。"""

        raise RuntimeError("scope service unavailable")


class SequenceClient:
    """按顺序提供完整图执行所需的模型响应。"""

    def __init__(self) -> None:
        """准备范围、规划、判定与回答四阶段响应。"""

        self.responses = [
            LLMResult(
                text=json.dumps(
                    {
                        "category": "enterprise_knowledge",
                        "reason": "enterprise investigation",
                    }
                ),
                model="fake-scope",
                request_id="scope-001",
                input_tokens=10,
                output_tokens=5,
            ),
            LLMResult(
                text=json.dumps(
                    {
                        "kind": "search_enterprise_knowledge",
                        "query": "project lifecycle archival",
                        "source": "jira",
                        "top_k": 5,
                        "reason": "Need enterprise evidence.",
                    }
                ),
                model="fake-planner",
                request_id="planner-001",
                input_tokens=20,
                output_tokens=10,
            ),
            LLMResult(
                text=json.dumps(
                    {
                        "decision": "supported",
                        "facts": [
                            {
                                "fact": "archival process",
                                "status": "supported",
                                "conclusion": "The process is documented.",
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "reason": "Evidence supports the answer.",
                    }
                ),
                model="fake-judge",
                request_id="judge-001",
                input_tokens=30,
                output_tokens=10,
            ),
            LLMResult(
                text="The archival process is documented [E1].",
                model="fake-answer",
                request_id="answer-001",
                input_tokens=40,
                output_tokens=10,
            ),
        ]

    def generate(self, **kwargs) -> LLMResult:
        """依次返回下一条预设模型响应。"""

        return self.responses.pop(0)


class FakeAdapter:
    """模拟企业搜索及证据解析适配器。"""

    def __init__(self) -> None:
        """初始化搜索与解析调用计数。"""

        self.search_calls = 0
        self.resolve_calls = 0

    def run(self, request) -> SearchToolResponse:
        """返回固定证据引用并记录一次搜索调用。"""

        self.search_calls += 1
        return SearchToolResponse(
            evidence=(
                EvidenceReference(
                    tenant_id="tenant-a",
                    source="jira",
                    document_id="document-001",
                    chunk_id="chunk-001",
                    sequence=0,
                    content_sha256="a" * 64,
                    acl_version="acl-v1",
                ),
            )
        )

    def resolve(
        self,
        references,
        runtime_context,
    ):
        """将证据引用解析为固定的搜索命中。"""

        self.resolve_calls += 1
        return (
            OpenSearchHit(
                score=1.0,
                chunk_id="chunk-001",
                document_id="document-001",
                source="jira",
                title="Project archival",
                sequence=0,
                text="The archival process is documented.",
                created_at=None,
                original_location=None,
            ),
        )


def make_runner(client) -> AgentRunner:
    """使用指定伪模型客户端构造测试运行器。"""

    return AgentRunner(
        llm_client=client,
        embedding_client=object(),
        opensearch_client=object(),
        hybrid_index_name="test-index",
    )


def test_write_request_is_rejected_without_llm() -> None:
    """验证写操作请求在调用模型前即被拒绝。"""

    result = make_runner(MustNotBeCalled()).ask(
        "请帮我删除 Jira INT-123",
        make_context(),
    )

    assert result.status == "scope_rejected"
    assert result.scope_category == "unsupported_action"
    assert result.tool_call_count == 0
    assert result.model_token_count == 0


def test_out_of_scope_request_stops_before_graph() -> None:
    """验证超出范围的请求不会进入图执行阶段。"""

    client = ScopeClient("out_of_scope")

    result = make_runner(client).ask(
        "Write a Snake game",
        make_context(),
    )

    assert result.status == "scope_rejected"
    assert result.scope_category == "out_of_scope"
    assert result.step_count == 0
    assert result.tool_call_count == 0
    assert client.calls == 1


def test_scope_failure_is_fail_closed() -> None:
    """验证范围分类失败时运行器以关闭方式报错。"""

    with pytest.raises(
        AgentRunnerUnavailable,
        match="系统暂时无法判断请求范围",
    ):
        make_runner(BrokenScopeClient()).ask(
            "Investigate project archival",
            make_context(),
        )


def test_enterprise_question_runs_full_graph(
    monkeypatch,
) -> None:
    """验证企业知识问题会完成检索、判定与回答流程。"""

    client = SequenceClient()
    adapter = FakeAdapter()

    monkeypatch.setattr(
        runner_module,
        "LocalEnterpriseSearchAdapter",
        lambda search_tool: adapter,
    )

    result = make_runner(client).ask(
        "How should the project be archived?",
        make_context(),
    )

    assert result.status == "completed"
    assert result.scope_category == "enterprise_knowledge"
    assert result.outcome == "supported"
    assert result.citations == ("E1",)
    assert result.step_count == 1
    assert result.tool_call_count == 1
    assert result.evidence_count == 1
    assert result.model_token_count == 135
    assert adapter.search_calls == 1
    assert adapter.resolve_calls == 2


def test_public_evidence_preserves_citation_order_and_uses_fresh_resolution(monkeypatch):
    class FreshAdapter(FakeAdapter):
        fresh_reads = 0
        def resolve(self, references, runtime_context, *, fresh=False):
            self.fresh_reads += int(fresh)
            return super().resolve(references, runtime_context)
    adapter = FreshAdapter()
    monkeypatch.setattr(runner_module, "LocalEnterpriseSearchAdapter", lambda tool: adapter)
    runner = make_runner(SequenceClient())
    runner._include_evidence = True
    result = runner.ask("How should the project be archived?", make_context())
    assert adapter.fresh_reads == 1
    assert result.evidence[0].id == "E1"
    assert result.evidence[0].chunk_id == "chunk-001"
    assert result.facts[0].evidence_ids == ["E1"]
    assert result.activity[0].status == "succeeded"


def test_failed_final_evidence_recheck_returns_no_answer(monkeypatch):
    class RevokedAdapter(FakeAdapter):
        def resolve(self, references, runtime_context, *, fresh=False):
            if fresh:
                raise RuntimeError("ACL revoked")
            return super().resolve(references, runtime_context)
    monkeypatch.setattr(runner_module, "LocalEnterpriseSearchAdapter", lambda tool: RevokedAdapter())
    runner = make_runner(SequenceClient())
    runner._include_evidence = True
    with pytest.raises(AgentRunnerUnavailable) as result:
        runner.ask("How should the project be archived?", make_context())
    assert result.value.error_code == "EVIDENCE_EXPORT_FAILED"
