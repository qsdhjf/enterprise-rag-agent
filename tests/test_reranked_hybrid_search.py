"""验证混合检索的重排、文档配额与失败回退行为。"""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_agent.infrastructure.reranker_client import (
    RerankerScore,
    RerankerServiceError,
)
from enterprise_agent.retrieval.hybrid_search import (
    hybrid_search_chunks,
)
from enterprise_agent.retrieval.keyword_search import OpenSearchHit


def _hit(number: int, *, document_id: str | None = None) -> OpenSearchHit:
    """构造带稳定排序信息的测试检索命中。"""

    return OpenSearchHit(
        score=float(20 - number),
        chunk_id=f"chunk-{number}",
        document_id=document_id or f"document-{number}",
        source="jira",
        title=f"Title {number}",
        sequence=number,
        text=f"Body {number}",
        created_at=None,
        original_location=None,
    )


@dataclass
class FakeReranker:
    """按预设分数返回重排结果并记录调用。"""

    scores: list[float]
    required: bool = True
    max_candidates: int = 20

    def __post_init__(self) -> None:
        """初始化重排调用记录。"""

        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def rerank(
        self,
        query: str,
        texts: list[str],
    ) -> tuple[RerankerScore, ...]:
        """记录查询与候选文本并返回预设分数。"""

        self.calls.append((query, tuple(texts)))
        return tuple(
            RerankerScore(index=index, score=score)
            for index, score in enumerate(self.scores[: len(texts)])
        )


def test_reranker_can_promote_candidate_outside_original_top_five(
    monkeypatch,
) -> None:
    """验证重排器可提升原始前五名之外的候选。"""

    bm25_hits = [_hit(number) for number in range(1, 11)]
    reranker = FakeReranker(
        scores=[0.0] * 7 + [10.0] + [0.0] * 12
    )

    monkeypatch.setattr(
        "enterprise_agent.retrieval.hybrid_search.search_chunks",
        lambda **kwargs: bm25_hits,
    )
    monkeypatch.setattr(
        "enterprise_agent.retrieval.hybrid_search.vector_search_chunks",
        lambda **kwargs: [],
    )

    results = hybrid_search_chunks(
        client=object(),
        index_name="index",
        embedding_client=object(),
        lexical_query="rewritten lexical query",
        semantic_query="original user question",
        top_k=5,
        reranker_client=reranker,
    )

    assert results[0].hit.chunk_id == "chunk-8"
    assert results[0].reranker_score == 10.0
    assert reranker.calls[0][0] == "original user question"
    assert "Title 8" in reranker.calls[0][1][7]
    assert "Body 8" in reranker.calls[0][1][7]


def test_document_quota_is_applied_after_reranking(monkeypatch) -> None:
    """验证单文档配额在重排完成后应用。"""

    bm25_hits = [
        _hit(1, document_id="same-document"),
        _hit(2, document_id="same-document"),
        _hit(3, document_id="same-document"),
        _hit(4),
    ]
    reranker = FakeReranker(scores=[4.0, 3.0, 2.0, 1.0])

    monkeypatch.setattr(
        "enterprise_agent.retrieval.hybrid_search.search_chunks",
        lambda **kwargs: bm25_hits,
    )
    monkeypatch.setattr(
        "enterprise_agent.retrieval.hybrid_search.vector_search_chunks",
        lambda **kwargs: [],
    )

    results = hybrid_search_chunks(
        client=object(),
        index_name="index",
        embedding_client=object(),
        lexical_query="query",
        top_k=3,
        max_chunks_per_document=2,
        reranker_client=reranker,
    )

    assert [result.hit.chunk_id for result in results] == [
        "chunk-1",
        "chunk-2",
        "chunk-4",
    ]


def test_optional_reranker_failure_falls_back_to_rrf(monkeypatch) -> None:
    """验证可选重排服务失败时回退到 RRF 排序。"""

    class FailingReranker:
        """模拟不可用的非必需重排服务。"""

        max_candidates = 20
        required = False

        def rerank(self, query, texts):
            """始终抛出服务不可用错误。"""

            raise RerankerServiceError("service unavailable")

    bm25_hits = [_hit(number) for number in range(1, 7)]

    monkeypatch.setattr(
        "enterprise_agent.retrieval.hybrid_search.search_chunks",
        lambda **kwargs: bm25_hits,
    )
    monkeypatch.setattr(
        "enterprise_agent.retrieval.hybrid_search.vector_search_chunks",
        lambda **kwargs: [],
    )

    results = hybrid_search_chunks(
        client=object(),
        index_name="index",
        embedding_client=object(),
        lexical_query="query",
        top_k=3,
        reranker_client=FailingReranker(),
    )

    assert [result.hit.chunk_id for result in results] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert all(result.reranker_score is None for result in results)
