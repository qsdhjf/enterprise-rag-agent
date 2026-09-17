"""验证重排服务的容量就绪检查。"""

from __future__ import annotations

import httpx
import pytest

from enterprise_agent.infrastructure.config import RerankerSettings
from enterprise_agent.infrastructure.reranker_client import TEIRerankerClient


def _client(*, capacity: int) -> TEIRerankerClient:
    """构造返回指定并发容量的测试重排客户端。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟重排服务的健康与信息端点。"""

        if request.url.path == "/health":
            return httpx.Response(200, text="ok")
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={
                    "model_id": "BAAI/bge-reranker-v2-m3",
                    "model_type": {"reranker": {}},
                    "max_concurrent_requests": capacity,
                    "max_batch_requests": capacity,
                    "max_client_batch_size": capacity,
                },
            )
        return httpx.Response(404)

    settings = RerankerSettings(
        base_url="http://reranker.test",
        model="BAAI/bge-reranker-v2-m3",
        timeout_seconds=5.0,
        max_candidates=40,
        required=True,
    )
    return TEIRerankerClient(
        settings,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("capacity, expected", [(40, True), (39, False)])
def test_readiness_requires_candidate_capacity(
    capacity: int,
    expected: bool,
) -> None:
    """验证就绪检查要求服务容量覆盖候选上限。"""

    assert _client(capacity=capacity).is_ready() is expected
