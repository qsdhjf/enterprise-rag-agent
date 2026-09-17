"""验证 TEI 重排客户端的请求契约与输入输出校验。"""

from __future__ import annotations

import httpx
import pytest

from enterprise_agent.infrastructure.config import RerankerSettings
from enterprise_agent.infrastructure.reranker_client import (
    RerankerServiceError,
    TEIRerankerClient,
)


def _settings(*, max_candidates: int = 40) -> RerankerSettings:
    """构造指向测试服务的重排器配置。"""

    return RerankerSettings(
        base_url="http://reranker.test",
        model="BAAI/bge-reranker-v2-m3",
        timeout_seconds=5.0,
        max_candidates=max_candidates,
        required=True,
    )


def test_rerank_uses_official_tei_contract() -> None:
    """验证重排请求遵循 TEI 官方接口契约。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """校验请求体并返回固定的 TEI 响应。"""

        assert request.url == "http://reranker.test/rerank"
        assert request.method == "POST"
        assert request.read().decode("utf-8") == (
            '{"query":"query","texts":["first","second"],'
            '"truncate":true,"truncation_direction":"right",'
            '"raw_scores":true,"return_text":false}'
        )
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 4.25},
                {"index": 0, "score": -0.5},
            ],
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )
    client = TEIRerankerClient(
        _settings(),
        http_client=http_client,
    )

    result = client.rerank(
        " query ",
        [" first ", "second"],
    )

    assert [(item.index, item.score) for item in result] == [
        (1, 4.25),
        (0, -0.5),
    ]


def test_rerank_rejects_duplicate_response_indices() -> None:
    """验证客户端拒绝响应中的重复候选索引。"""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=[
                {"index": 0, "score": 1.0},
                {"index": 0, "score": 0.5},
            ],
        )
    )
    client = TEIRerankerClient(
        _settings(),
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(
        RerankerServiceError,
        match="duplicate index",
    ):
        client.rerank("query", ["first", "second"])


def test_rerank_enforces_candidate_budget_before_http() -> None:
    """验证候选数超限时在发起 HTTP 请求前失败。"""

    client = TEIRerankerClient(
        _settings(max_candidates=1),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: pytest.fail(
                    "HTTP must not be called"
                )
            )
        ),
    )

    with pytest.raises(ValueError, match="configured maximum"):
        client.rerank("query", ["first", "second"])
