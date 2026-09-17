"""验证 FastAPI 对 AgentRunner 的 HTTP 适配行为。

通过替换应用生命周期和运行器，覆盖正常问答、范围拒绝、服务不可用、请求校验和就绪探针。
"""

from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi.testclient import TestClient

import enterprise_agent.api as api_module
from enterprise_agent.agent.runner import (
    AgentReadiness,
    AgentRunResult,
    AgentRunnerUnavailable,
)


class FakeRunner:
    """模拟运行器的就绪检查与多种问答结果。"""

    def __init__(self, mode: str = "completed") -> None:
        """设置响应模式并初始化请求记录。"""

        self.mode = mode
        self.calls = 0
        self.last_request_id: str | None = None

    def check_readiness(self) -> AgentReadiness:
        """返回各依赖均已就绪的固定检查结果。"""

        return AgentReadiness(
            ready=True,
            opensearch_reachable=True,
            hybrid_index_exists=True,
            indexed_chunks=780,
            message="ready",
        )

    def ask(self, question, runtime_context):
        """根据响应模式返回结果或模拟服务不可用。"""

        self.calls += 1
        self.last_request_id = runtime_context.request_id

        if self.mode == "unavailable":
            raise AgentRunnerUnavailable(
                "TEST_UNAVAILABLE",
                "temporary failure",
            )

        if self.mode == "scope_rejected":
            return AgentRunResult(
                status="scope_rejected",
                request_id=runtime_context.request_id,
                task_id=None,
                question=question,
                scope_category="out_of_scope",
                outcome=None,
                answer="当前只处理企业知识查询。",
                citations=(),
                step_count=0,
                tool_call_count=0,
                evidence_count=0,
                model_token_count=15,
            )

        return AgentRunResult(
            status="completed",
            request_id=runtime_context.request_id,
            task_id="task-001",
            question=question,
            scope_category="enterprise_knowledge",
            outcome="supported",
            answer="The process is documented [E1].",
            citations=("E1",),
            step_count=1,
            tool_call_count=1,
            evidence_count=1,
            model_token_count=135,
        )


def make_client(monkeypatch, runner: FakeRunner) -> TestClient:
    """创建在应用生命周期中注入伪运行器的测试客户端。"""

    @asynccontextmanager
    async def fake_lifespan(app):
        """在测试生命周期内向应用状态注入伪运行器。"""

        app.state.agent_runner = runner
        app.state.demo_mode = True
        app.state.request_slots = BoundedSemaphore(2)
        yield

    monkeypatch.setattr(
        api_module.app.router,
        "lifespan_context",
        fake_lifespan,
    )
    return TestClient(api_module.app)


def test_ask_returns_langgraph_result(monkeypatch) -> None:
    """验证问答接口会返回完整的图执行结果与请求标识。"""

    runner = FakeRunner()

    with make_client(monkeypatch, runner) as client:
        response = client.post(
            "/ask",
            json={"question": "How is the project archived?"},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "completed"
    assert body["outcome"] == "supported"
    assert body["citations"] == ["E1"]
    assert body["task_id"] == "task-001"
    assert body["request_id"] == runner.last_request_id
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_scope_rejection_is_normal_response(monkeypatch) -> None:
    """验证范围拒绝会作为正常业务响应返回。"""

    runner = FakeRunner("scope_rejected")

    with make_client(monkeypatch, runner) as client:
        response = client.post(
            "/ask",
            json={"question": "Write a Snake game"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "scope_rejected"
    assert body["scope_category"] == "out_of_scope"
    assert body["tool_call_count"] == 0


def test_agent_unavailable_returns_503(monkeypatch) -> None:
    """验证运行器不可用时接口返回服务不可用状态。"""

    runner = FakeRunner("unavailable")

    with make_client(monkeypatch, runner) as client:
        response = client.post(
            "/ask",
            json={"question": "Investigate project archival"},
        )

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == (
        "Agent 暂时无法完成请求"
    )
    assert "X-Request-ID" in response.headers


def test_extra_request_field_is_rejected(monkeypatch) -> None:
    """验证请求包含未声明字段时会被参数校验拒绝。"""

    runner = FakeRunner()

    with make_client(monkeypatch, runner) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Investigate project archival",
                "top_k": 100,
            },
        )

    assert response.status_code == 422
    assert runner.calls == 0


def test_readiness_uses_agent_runner(monkeypatch) -> None:
    """验证就绪探针返回运行器提供的检查信息。"""

    runner = FakeRunner()

    with make_client(monkeypatch, runner) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["indexed_chunks"] == 780
