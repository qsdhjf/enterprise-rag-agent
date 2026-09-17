"""验证发送给规划模型的结构化输入。

用例对比首次规划与重试规划，确保只有重试时才携带历史动作和待补事实。
"""

import json

from enterprise_agent.agent.contracts import (
    AgentRuntimeContext,
    PrincipalContext,
    RetrievalPolicy,
    RunBudget,
)
from enterprise_agent.agent.state import (
    EvidenceAssessmentState,
    EvidenceFactState,
    EvidenceReference,
    ToolCallRecord,
    create_initial_graph_state,
)
from enterprise_agent.agent.planner import (
    _build_planner_input,
    plan_next_action,
)
from enterprise_agent.infrastructure.llm_client import (
    LLMResult,
)


class SequencePlannerClient:
    """按顺序返回预设结果，用于验证 Planner 的有限重试。"""

    def __init__(
        self,
        responses: list[LLMResult | Exception],
    ) -> None:
        """保存响应或异常序列，并初始化请求记录。"""
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def generate(
        self,
        **kwargs: object,
    ) -> LLMResult:
        """记录请求并按调用顺序返回结果，预设异常则原样抛出。"""
        self.requests.append(kwargs)
        response = self.responses[
            len(self.requests) - 1
        ]

        if isinstance(response, Exception):
            raise response

        return response


def make_context() -> AgentRuntimeContext:
    """构造仅允许 Jira 检索的规划器测试运行上下文。"""
    return AgentRuntimeContext(
        request_id="planner-input-test-001",
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


def test_first_planning_input_has_no_history() -> None:
    """验证首轮规划输入没有证据结论、缺失事实或历史搜索。"""
    context = make_context()

    state = create_initial_graph_state(
        goal="Find the deployment date.",
        runtime_context=context,
    )

    planner_input = _build_planner_input(
        state
    )

    assert planner_input["step_number"] == 1
    assert planner_input["evidence_count"] == 0
    assert (
        planner_input["evidence_decision"]
        is None
    )
    assert planner_input["missing_facts"] == []
    assert planner_input["previous_searches"] == []


def test_retry_input_contains_missing_facts() -> None:
    """验证重试规划输入会汇总缺失事实和上一轮搜索记录。"""
    context = make_context()

    state = create_initial_graph_state(
        goal=(
            "Find the deployment date "
            "and approved change ticket."
        ),
        runtime_context=context,
    )

    state["step_count"] = 1

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

    state["evidence_assessment"] = (
        EvidenceAssessmentState(
            decision="partially_supported",
            facts=[
                EvidenceFactState(
                    fact="deployment date",
                    status="supported",
                    conclusion="2026-08-01",
                    evidence_ids=["E1"],
                ),
                EvidenceFactState(
                    fact="approved change-ticket ID",
                    status="missing",
                    conclusion=None,
                    evidence_ids=[],
                ),
            ],
            reason=(
                "Deployment date found, "
                "but ticket ID is missing."
            ),
            judge_succeeded=True,
            used_model=True,
            error_code=None,
            model="fake-judge",
            input_tokens=100,
            output_tokens=30,
            request_id="judge-001",
            assessed_chunk_ids=["chunk-001"],
            assessed_at=(
                "2026-08-14T00:00:00+00:00"
            ),
        )
    )

    state["tool_calls"] = [
        ToolCallRecord(
            call_id="search-001",
            step_number=1,
            tool_name=(
                "search_enterprise_knowledge"
            ),
            status="succeeded",
            query="project deployment date",
            source="jira",
            top_k=5,
            returned_hits=1,
            new_evidence_count=1,
            started_at=(
                "2026-08-14T00:00:00+00:00"
            ),
            completed_at=(
                "2026-08-14T00:00:01+00:00"
            ),
            error_code=None,
        )
    ]

    planner_input = _build_planner_input(
        state
    )

    assert planner_input["step_number"] == 2
    assert planner_input["evidence_count"] == 1

    assert (
        planner_input["evidence_decision"]
        == "partially_supported"
    )

    assert planner_input["missing_facts"] == [
        "approved change-ticket ID"
    ]

    assert planner_input["previous_searches"] == [
        {
            "query": "project deployment date",
            "source": "jira",
            "status": "succeeded",
            "returned_hits": 1,
            "new_evidence_count": 1,
        }
    ]


def test_invalid_output_is_repaired_once() -> None:
    """验证首次非 JSON 输出会触发一次修复并产生合法搜索动作。"""

    state = create_initial_graph_state(
        goal="Find the project archival process.",
        runtime_context=make_context(),
    )
    client = SequencePlannerClient(
        [
            LLMResult(
                text="I recommend searching Jira first.",
                model="fake-planner",
                request_id="planner-invalid-001",
                input_tokens=80,
                output_tokens=20,
            ),
            LLMResult(
                text=json.dumps(
                    {
                        "kind": (
                            "search_enterprise_knowledge"
                        ),
                        "query": (
                            "project lifecycle archival"
                        ),
                        "source": "jira",
                        "top_k": 5,
                        "reason": (
                            "Need enterprise evidence."
                        ),
                    }
                ),
                model="fake-planner",
                request_id="planner-repaired-002",
                input_tokens=100,
                output_tokens=30,
            ),
        ]
    )

    updates = plan_next_action(
        state,
        client,
    )

    assert len(client.requests) == 2
    assert client.requests[0]["response_format"] == {
        "type": "json_object",
    }
    assert client.requests[1]["response_format"] == {
        "type": "json_object",
    }
    assert updates["step_count"] == 1
    assert updates["model_token_count"] == 230
    assert updates["last_error_code"] is None

    action = updates["pending_action"]
    assert action is not None
    assert (
        action["kind"]
        == "search_enterprise_knowledge"
    )
    assert action["source"] == "jira"

    repair_request = client.requests[1]
    repair_payload = json.loads(
        str(repair_request["user_prompt"])
    )
    assert (
        repair_payload["previous_response"]
        == "I recommend searching Jira first."
    )
    assert "validation_error" in repair_payload
    assert (
        "Repair mode"
        in str(repair_request["system_prompt"])
    )


def test_second_invalid_output_fails_closed() -> None:
    """验证修复后仍不合法时停止，并且不生成任何待执行动作。"""

    state = create_initial_graph_state(
        goal="Find the project archival process.",
        runtime_context=make_context(),
    )
    client = SequencePlannerClient(
        [
            LLMResult(
                text="not json",
                model="fake-planner",
                request_id="planner-invalid-001",
                input_tokens=40,
                output_tokens=10,
            ),
            LLMResult(
                text='{"kind": "finish"}',
                model="fake-planner",
                request_id="planner-invalid-002",
                input_tokens=50,
                output_tokens=10,
            ),
        ]
    )

    updates = plan_next_action(
        state,
        client,
    )

    assert len(client.requests) == 2
    assert updates["status"] == "failed"
    assert updates["step_count"] == 1
    assert updates["model_token_count"] == 110
    assert updates["pending_action"] is None
    assert updates["approved_action"] is None
    assert (
        updates["last_error_code"]
        == "INVALID_PLANNER_OUTPUT"
    )


def test_repair_model_failure_stops_without_action() -> None:
    """验证修复调用本身失败时仍然安全停止。"""

    state = create_initial_graph_state(
        goal="Find the project archival process.",
        runtime_context=make_context(),
    )
    client = SequencePlannerClient(
        [
            LLMResult(
                text="not json",
                model="fake-planner",
                request_id="planner-invalid-001",
                input_tokens=40,
                output_tokens=10,
            ),
            RuntimeError("repair unavailable"),
        ]
    )

    updates = plan_next_action(
        state,
        client,
    )

    assert len(client.requests) == 2
    assert updates["status"] == "failed"
    assert updates["model_token_count"] == 50
    assert updates["pending_action"] is None
    assert (
        updates["last_error_code"]
        == "PLANNER_MODEL_FAILED"
    )


def test_repair_respects_remaining_token_budget() -> None:
    """验证剩余 Token 不足时不会发起第二次模型调用。"""

    base_context = make_context()
    context = AgentRuntimeContext(
        request_id="planner-budget-test-001",
        principal=base_context.principal,
        budget=RunBudget(
            max_model_tokens=1600,
        ),
        retrieval_policy=(
            base_context.retrieval_policy
        ),
    )
    state = create_initial_graph_state(
        goal="Find the project archival process.",
        runtime_context=context,
    )
    client = SequencePlannerClient(
        [
            LLMResult(
                text="not json",
                model="fake-planner",
                request_id="planner-invalid-001",
                input_tokens=100,
                output_tokens=100,
            ),
        ]
    )

    updates = plan_next_action(
        state,
        client,
    )

    assert len(client.requests) == 1
    assert updates["status"] == "budget_exceeded"
    assert updates["step_count"] == 1
    assert updates["model_token_count"] == 200
    assert updates["pending_action"] is None
    assert (
        updates["last_error_code"]
        == "INSUFFICIENT_MODEL_TOKEN_BUDGET"
    )


def test_valid_output_is_discarded_after_token_overrun() -> None:
    """验证首次响应实耗超预算时，即使 JSON 合法也不能生成动作。"""

    base_context = make_context()
    context = AgentRuntimeContext(
        request_id="planner-overrun-test-001",
        principal=base_context.principal,
        budget=RunBudget(
            max_model_tokens=1600,
        ),
        retrieval_policy=(
            base_context.retrieval_policy
        ),
    )
    state = create_initial_graph_state(
        goal="Find the project archival process.",
        runtime_context=context,
    )
    client = SequencePlannerClient(
        [
            LLMResult(
                text=json.dumps(
                    {
                        "kind": (
                            "search_enterprise_knowledge"
                        ),
                        "query": "project archival",
                        "source": "jira",
                        "top_k": 5,
                        "reason": "Need evidence.",
                    }
                ),
                model="fake-planner",
                request_id="planner-overrun-001",
                input_tokens=1600,
                output_tokens=100,
            ),
        ]
    )

    updates = plan_next_action(
        state,
        client,
    )

    assert len(client.requests) == 1
    assert updates["status"] == "budget_exceeded"
    assert updates["model_token_count"] == 1700
    assert updates["pending_action"] is None
    assert (
        updates["last_error_code"]
        == "MODEL_TOKEN_BUDGET_EXCEEDED"
    )


def test_repaired_output_is_discarded_after_token_overrun() -> None:
    """验证两次调用累计超预算时，修复成功的动作仍必须丢弃。"""

    base_context = make_context()
    context = AgentRuntimeContext(
        request_id="planner-repair-overrun-test-001",
        principal=base_context.principal,
        budget=RunBudget(
            max_model_tokens=2200,
        ),
        retrieval_policy=(
            base_context.retrieval_policy
        ),
    )
    state = create_initial_graph_state(
        goal="Find the project archival process.",
        runtime_context=context,
    )
    client = SequencePlannerClient(
        [
            LLMResult(
                text="not json",
                model="fake-planner",
                request_id="planner-invalid-001",
                input_tokens=500,
                output_tokens=100,
            ),
            LLMResult(
                text=json.dumps(
                    {
                        "kind": "finish",
                        "reason": "Evidence is sufficient.",
                    }
                ),
                model="fake-planner",
                request_id="planner-repaired-002",
                input_tokens=1600,
                output_tokens=100,
            ),
        ]
    )

    updates = plan_next_action(
        state,
        client,
    )

    assert len(client.requests) == 2
    assert updates["status"] == "budget_exceeded"
    assert updates["model_token_count"] == 2300
    assert updates["pending_action"] is None
    assert (
        updates["last_error_code"]
        == "MODEL_TOKEN_BUDGET_EXCEEDED"
    )
