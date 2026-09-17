"""同时使用 BM25 与向量召回，再通过倒数排名融合生成统一结果。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from opensearchpy import OpenSearch

from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.infrastructure.reranker_client import (
    RerankerServiceError,
    TEIRerankerClient,
)
from enterprise_agent.retrieval.access_control import (
    RetrievalAccessContext,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
    search_chunks,
)
from enterprise_agent.retrieval.vector_search import (
    vector_search_chunks,
)


@dataclass(frozen=True, slots=True)
class HybridSearchHit:
    """融合后的命中项及其两条召回路径的排名、原始分数。"""

    hit: OpenSearchHit
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    vector_rank: int | None
    vector_score: float | None
    reranker_rank: int | None = None
    reranker_score: float | None = None


def _first_hit_by_chunk_id(
    hits: Sequence[OpenSearchHit],
) -> dict[str, tuple[int, OpenSearchHit]]:
    """按块 ID 去重，并记录每个块首次出现的从 1 开始排名。"""

    result: dict[str, tuple[int, OpenSearchHit]] = {}

    for rank, hit in enumerate(hits, start=1):
        # 只保留首次出现的位置，因为它代表该列表中的最佳排名。
        if hit.chunk_id not in result:
            result[hit.chunk_id] = (rank, hit)

    return result


def reciprocal_rank_fuse(
    bm25_hits: Sequence[OpenSearchHit],
    vector_hits: Sequence[OpenSearchHit],
    top_k: int = 5,
    rrf_k: int = 60,
    max_chunks_per_document: int = 2,
) -> list[HybridSearchHit]:
    """使用 Reciprocal Rank Fusion（RRF）合并两组有序命中。

    每条路径对块的贡献为 ``1 / (rrf_k + rank)``。该算法只依赖排名，
    因而不需要直接比较 BM25 分数和向量相似度这两种不同量纲的数值。
    融合完成后再执行文档级配额，并截取 ``top_k``。
    """

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    if max_chunks_per_document <= 0:
        raise ValueError(
            "max_chunks_per_document 必须大于 0"
        )

    bm25_by_id = _first_hit_by_chunk_id(bm25_hits)
    vector_by_id = _first_hit_by_chunk_id(vector_hits)

    # 取并集：只在任一路径召回的块也有资格参与最终排序。
    chunk_ids = set(bm25_by_id) | set(vector_by_id)
    fused_hits: list[HybridSearchHit] = []

    for chunk_id in sorted(chunk_ids):
        bm25_entry = bm25_by_id.get(chunk_id)
        vector_entry = vector_by_id.get(chunk_id)

        bm25_rank = (
            bm25_entry[0]
            if bm25_entry is not None
            else None
        )
        vector_rank = (
            vector_entry[0]
            if vector_entry is not None
            else None
        )

        # 两路命中共享业务字段；优先复用 BM25 对象，否则使用向量命中。
        base_hit = (
            bm25_entry[1]
            if bm25_entry is not None
            else vector_entry[1]
        )

        if bm25_entry and vector_entry:
            bm25_hit = bm25_entry[1]
            vector_hit = vector_entry[1]

            # 同一 chunk_id 指向不同文档意味着索引或上游数据已不一致，不能静默融合。
            if (
                bm25_hit.document_id
                != vector_hit.document_id
                or bm25_hit.source != vector_hit.source
            ):
                raise RuntimeError(
                    f"同一切块的数据不一致：{chunk_id}"
                )

        rrf_score = 0.0

        if bm25_rank is not None:
            rrf_score += 1.0 / (rrf_k + bm25_rank)

        if vector_rank is not None:
            rrf_score += 1.0 / (
                rrf_k + vector_rank
            )

        fused_hits.append(
            HybridSearchHit(
                hit=base_hit,
                rrf_score=rrf_score,
                bm25_rank=bm25_rank,
                bm25_score=(
                    bm25_entry[1].score
                    if bm25_entry
                    else None
                ),
                vector_rank=vector_rank,
                vector_score=(
                    vector_entry[1].score
                    if vector_entry
                    else None
                ),
            )
        )

    def sort_key(
        result: HybridSearchHit,
    ) -> tuple:
        """生成确定性排序键：融合分相同时优先双路命中和更靠前的排名。"""

        ranks = [
            rank
            for rank in (
                result.bm25_rank,
                result.vector_rank,
            )
            if rank is not None
        ]

        return (
            -result.rrf_score,
            -len(ranks),
            min(ranks),
            sum(ranks),
            result.hit.chunk_id,
        )

    fused_hits.sort(key=sort_key)

    final_results: list[HybridSearchHit] = []
    document_counts: dict[str, int] = {}

    for result in fused_hits:
        document_id = result.hit.document_id
        current_count = document_counts.get(
            document_id,
            0,
        )

        # 最后阶段限制同一文档的块数，以增加证据来源多样性。
        if current_count >= max_chunks_per_document:
            continue

        document_counts[document_id] = current_count + 1
        final_results.append(result)

        if len(final_results) >= top_k:
            break

    return final_results
def _limit_hybrid_results(
    results: Sequence[HybridSearchHit],
    top_k: int,
    max_chunks_per_document: int,
) -> list[HybridSearchHit]:
    """在全部排序阶段结束后执行文档级配额和数量截断。"""

    final_results: list[HybridSearchHit] = []
    document_counts: dict[str, int] = {}

    for result in results:
        document_id = result.hit.document_id
        current_count = document_counts.get(document_id, 0)

        if current_count >= max_chunks_per_document:
            continue

        document_counts[document_id] = current_count + 1
        final_results.append(result)

        if len(final_results) >= top_k:
            break

    return final_results



def hybrid_search_chunks(
    client: OpenSearch,
    index_name: str,
    embedding_client: OpenAICompatibleEmbeddingClient,
    lexical_query: str,
    semantic_query: str | None = None,
    top_k: int = 5,
    source: str | None = None,
    max_chunks_per_document: int = 2,
    candidate_multiplier: int = 4,
    rrf_k: int = 60,
    reranker_client: TEIRerankerClient | None = None,
    access_context: RetrievalAccessContext | None = None,
) -> list[HybridSearchHit]:
    """分别执行关键词和语义召回，然后以 RRF 合并候选。

    ``lexical_query`` 可使用经过改写的中英文关键词；``semantic_query`` 通常保留
    用户原始问题。每路先扩大候选池，再统一执行最终的文档配额和 top-k 截断。
    """

    lexical_query = lexical_query.strip()

    if not lexical_query:
        raise ValueError("lexical_query 不能为空")

    semantic_query = (
        semantic_query.strip()
        if semantic_query is not None
        else lexical_query
    )

    if not semantic_query:
        raise ValueError("semantic_query 不能为空")

    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100 之间")

    if candidate_multiplier <= 0:
        raise ValueError(
            "candidate_multiplier 必须大于 0"
        )

    # 至少召回 20 条、最多 100 条，为两路融合和文档去重保留余量。
    candidate_k = min(
        max(top_k * candidate_multiplier, 20),
        100,
    )
    effective_access = (
        access_context or RetrievalAccessContext.local_demo()
    )

    # 两条召回路径使用相同来源过滤和候选规模，但使用各自适合的查询文本。
    bm25_hits = search_chunks(
        client=client,
        index_name=index_name,
        query=lexical_query,
        top_k=candidate_k,
        source=source,
        max_chunks_per_document=candidate_k,
        access_context=effective_access,
    )

    vector_hits = vector_search_chunks(
        client=client,
        index_name=index_name,
        embedding_client=embedding_client,
        query=semantic_query,
        top_k=candidate_k,
        source=source,
        max_chunks_per_document=candidate_k,
        access_context=effective_access,
    )

    if reranker_client is None:
        return reciprocal_rank_fuse(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            top_k=top_k,
            rrf_k=rrf_k,
            max_chunks_per_document=max_chunks_per_document,
        )

    if reranker_client.max_candidates < top_k:
        raise ValueError(
            "reranker max_candidates must be greater than or equal to top_k: "
            f"{reranker_client.max_candidates} < {top_k}"
        )

    rerank_candidate_k = min(
        candidate_k,
        reranker_client.max_candidates,
    )
    rrf_candidates = reciprocal_rank_fuse(
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
        top_k=rerank_candidate_k,
        rrf_k=rrf_k,
        # Keep all candidate chunks until reranking has finished.
        max_chunks_per_document=rerank_candidate_k,
    )

    if not rrf_candidates:
        return []

    passages = [
        f"Title: {result.hit.title}\n\nText: {result.hit.text}"
        for result in rrf_candidates
    ]

    try:
        reranker_scores = reranker_client.rerank(
            semantic_query,
            passages,
        )
    except RerankerServiceError:
        if reranker_client.required:
            raise

        return reciprocal_rank_fuse(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            top_k=top_k,
            rrf_k=rrf_k,
            max_chunks_per_document=max_chunks_per_document,
        )

    expected_indices = set(range(len(rrf_candidates)))
    score_by_index = {
        score.index: score.score
        for score in reranker_scores
    }

    if (
        len(reranker_scores) != len(rrf_candidates)
        or set(score_by_index) != expected_indices
    ):
        raise RuntimeError(
            "reranker result indexes do not match candidate indexes"
        )

    scored_candidates = [
        (
            original_rank,
            replace(
                result,
                reranker_score=score_by_index[original_rank],
            ),
        )
        for original_rank, result in enumerate(rrf_candidates)
    ]
    scored_candidates.sort(
        key=lambda item: (
            -score_by_index[item[0]],
            -item[1].rrf_score,
            item[0],
            item[1].hit.chunk_id,
        )
    )

    reranked = [
        replace(result, reranker_rank=rank)
        for rank, (_, result) in enumerate(
            scored_candidates,
            start=1,
        )
    ]

    return _limit_hybrid_results(
        reranked,
        top_k=top_k,
        max_chunks_per_document=max_chunks_per_document,
    )
