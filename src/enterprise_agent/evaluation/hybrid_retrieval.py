"""在同一评测集上对比 BM25、纯向量和 RRF 混合检索的质量与延迟。"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from enterprise_agent.infrastructure.config import (
    load_embedding_settings,
    load_llm_settings,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.retrieval.hybrid_search import (
    reciprocal_rank_fuse,
)
from enterprise_agent.retrieval.keyword_search import (
    OpenSearchHit,
    search_chunks,
)
from enterprise_agent.retrieval.vector_search import (
    vector_search_chunks,
)
from enterprise_agent.retrieval.query_rewriter import rewrite_query
from enterprise_agent.evaluation.retrieval import load_cases


# 三种方法和统一的截断位置用于构造结构一致的统计字典。
METHODS = ("bm25", "vector", "hybrid")
KS = (1, 3, 5)


def elapsed_ms(started_at: float) -> float:
    """返回从给定高精度计时起点到现在的毫秒数。"""

    return (perf_counter() - started_at) * 1000


def unique_document_ids(
    hits: list[OpenSearchHit],
    limit: int = 5,
) -> list[str]:
    """保持命中顺序提取不重复的文档 ID，最多返回 ``limit`` 个。"""

    document_ids: list[str] = []
    seen: set[str] = set()

    for hit in hits:
        if hit.document_id in seen:
            continue

        seen.add(hit.document_id)
        document_ids.append(hit.document_id)

        if len(document_ids) >= limit:
            break

    return document_ids


def first_relevant_rank(
    returned_ids: list[str],
    relevant_ids: frozenset[str],
) -> int | None:
    """返回首篇标准相关文档的从 1 开始排名，未命中时返回 ``None``。"""

    for rank, document_id in enumerate(
        returned_ids,
        start=1,
    ):
        if document_id in relevant_ids:
            return rank

    return None


def rank_text(rank: int | None) -> str:
    """把可选排名转换为适合命令行报告的中文文字。"""

    if rank is None:
        return "前5名未命中"

    return str(rank)


def validate_index(
    client,
    index_name: str,
    expected_count: int,
    cases,
) -> None:
    """在正式评测前校验索引块总数及所有标准相关文档是否存在。

    该预检防止因索引构建不完整而把数据问题误判为检索算法质量下降。
    """

    actual_count = client.count(
        index=index_name
    )["count"]

    print(f"评测索引：{index_name}")
    print(f"索引切块数量：{actual_count}")

    if actual_count != expected_count:
        raise RuntimeError(
            "索引切块数量不符合预期："
            f"期望 {expected_count}，实际 {actual_count}"
        )

    # 合并所有用例的标准答案，避免对同一文档重复执行存在性查询。
    expected_document_ids = sorted(
        {
            document_id
            for case in cases
            for document_id in case.relevant_document_ids
        }
    )

    missing_ids: list[str] = []

    for document_id in expected_document_ids:
        count = client.count(
            index=index_name,
            body={
                "query": {
                    "term": {
                        "document_id": document_id,
                    }
                }
            },
        )["count"]

        if count == 0:
            missing_ids.append(document_id)

    if missing_ids:
        raise RuntimeError(
            "以下标准文档不在评测索引中："
            + ", ".join(missing_ids)
        )

    print(
        f"标准文档预检：通过，"
        f"共 {len(expected_document_ids)} 个唯一文档"
    )


def main() -> None:
    """运行三路对照实验，并输出逐用例排名、Hit@K、MRR 与耗时统计。"""

    parser = argparse.ArgumentParser(
        description="对比BM25、向量和RRF混合检索"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSONL格式的人工标注评测集",
    )
    parser.add_argument(
        "--rewrite-query",
        action="store_true",
        help="为BM25自动生成中英检索词",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="每条检索路线参与融合的候选切块数",
    )
    parser.add_argument(
        "--expected-index-count",
        type=int,
        default=780,
        help="评测索引预期包含的切块数量",
    )
    args = parser.parse_args()

    if not 5 <= args.candidate_k <= 100:
        raise ValueError(
            "--candidate-k必须在5到100之间"
        )

    cases = load_cases(args.cases.resolve())

    opensearch_settings = load_opensearch_settings()
    embedding_settings = load_embedding_settings()

    opensearch_client = create_opensearch_client(
        opensearch_settings
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        embedding_settings
    )

    rewrite_client = None

    if args.rewrite_query:
        rewrite_client = OpenAICompatibleLLMClient(
            load_llm_settings()
        )

    # 为每种方法分别累计质量、延迟；融合自身耗时另记，便于拆分开销。
    hit_counts = {
        method: {k: 0 for k in KS}
        for method in METHODS
    }
    reciprocal_rank_totals = {
        method: 0.0
        for method in METHODS
    }
    latency_ms = {
        method: []
        for method in METHODS
    }
    fusion_latency_ms: list[float] = []

    rewrite_call_count = 0
    rewrite_input_tokens = 0
    rewrite_output_tokens = 0
    rewrite_latency_ms: list[float] = []

    try:
        if not opensearch_client.ping():
            raise RuntimeError("无法连接OpenSearch")

        validate_index(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            expected_count=args.expected_index_count,
            cases=cases,
        )

        # 不计入正式延迟：预热本地模型和搜索连接，减少首次调用冷启动偏差。
        print("正在预热本地BGE-M3和OpenSearch……")

        warmup_case = cases[0]

        search_chunks(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            query=warmup_case.query,
            top_k=1,
            source=warmup_case.source,
            max_chunks_per_document=1,
        )

        vector_search_chunks(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            embedding_client=embedding_client,
            query=warmup_case.query,
            top_k=1,
            source=warmup_case.source,
            max_chunks_per_document=1,
        )

        print("预热完成，开始正式评测。")

        for case in cases:
            lexical_query = case.query

            if rewrite_client is not None:
                # 只有确实发生中文改写时才计入模型调用与改写延迟。
                rewrite_started = perf_counter()

                rewrite_result = rewrite_query(
                    query=case.query,
                    client=rewrite_client,
                )

                if rewrite_result.rewritten:
                    rewrite_call_count += 1
                    rewrite_latency_ms.append(
                        elapsed_ms(rewrite_started)
                    )
                    rewrite_input_tokens += (
                        rewrite_result.input_tokens or 0
                    )
                    rewrite_output_tokens += (
                        rewrite_result.output_tokens or 0
                    )

                lexical_query = (
                    rewrite_result.retrieval_query
                )

            # 三个阶段独立计时，以区分关键词、向量服务和纯 Python 融合成本。
            bm25_started = perf_counter()

            bm25_hits = search_chunks(
                client=opensearch_client,
                index_name=(
                    opensearch_settings.hybrid_index_name
                ),
                query=lexical_query,
                top_k=args.candidate_k,
                source=case.source,
                max_chunks_per_document=(
                    args.candidate_k
                ),
            )

            bm25_ms = elapsed_ms(bm25_started)

            vector_started = perf_counter()

            vector_hits = vector_search_chunks(
                client=opensearch_client,
                index_name=(
                    opensearch_settings.hybrid_index_name
                ),
                embedding_client=embedding_client,
                query=case.query,
                top_k=args.candidate_k,
                source=case.source,
                max_chunks_per_document=(
                    args.candidate_k
                ),
            )

            vector_ms = elapsed_ms(vector_started)

            fusion_started = perf_counter()

            hybrid_hits = reciprocal_rank_fuse(
                bm25_hits=bm25_hits,
                vector_hits=vector_hits,
                top_k=5,
                max_chunks_per_document=1,
            )

            fusion_ms = elapsed_ms(fusion_started)

            # 统一转换成文档 ID 列表后再计算指标，避免块级重复影响评测。
            returned_ids = {
                "bm25": unique_document_ids(
                    bm25_hits,
                    limit=5,
                ),
                "vector": unique_document_ids(
                    vector_hits,
                    limit=5,
                ),
                "hybrid": [
                    result.hit.document_id
                    for result in hybrid_hits
                ],
            }

            current_latencies = {
                "bm25": bm25_ms,
                "vector": vector_ms,

                # 当前实现两路顺序执行，因此混合总耗时取三阶段之和。
                "hybrid": (
                    bm25_ms
                    + vector_ms
                    + fusion_ms
                ),
            }

            ranks: dict[str, int | None] = {}

            for method in METHODS:
                rank = first_relevant_rank(
                    returned_ids=returned_ids[method],
                    relevant_ids=(
                        case.relevant_document_ids
                    ),
                )
                ranks[method] = rank

                # 首个相关文档排名 <= K 即表示该用例在 Hit@K 上命中。
                for k in KS:
                    if rank is not None and rank <= k:
                        hit_counts[method][k] += 1

                reciprocal_rank_totals[method] += (
                    1.0 / rank
                    if rank is not None
                    else 0.0
                )

                latency_ms[method].append(
                    current_latencies[method]
                )

            fusion_latency_ms.append(fusion_ms)

            print()
            print(f"用例：{case.case_id}")
            print(f"查询：{case.query}")
            print(
                f"限定来源：{case.source or '全部'}"
            )

            if lexical_query != case.query:
                print(f"BM25检索词：{lexical_query}")

            print(
                "BM25："
                f"首个正确排名 {rank_text(ranks['bm25'])}，"
                f"耗时 {bm25_ms:.2f} ms"
            )
            print(
                "向量："
                f"首个正确排名 {rank_text(ranks['vector'])}，"
                f"耗时 {vector_ms:.2f} ms"
            )
            print(
                "混合："
                f"首个正确排名 {rank_text(ranks['hybrid'])}，"
                f"顺序总耗时 "
                f"{current_latencies['hybrid']:.2f} ms"
            )

    finally:
        # 任一路评测失败都要释放模型 HTTP 与 OpenSearch 连接。
        embedding_client.close()
        opensearch_client.close()

    case_count = len(cases)

    print()
    print("========== 三路总体指标 ==========")
    print(f"用例数量：{case_count}")

    labels = {
        "bm25": "BM25",
        "vector": "纯向量",
        "hybrid": "RRF混合",
    }

    for method in METHODS:
        print()
        print(f"[{labels[method]}]")

        for k in KS:
            hit_rate = (
                hit_counts[method][k]
                / case_count
            )
            print(f"Hit@{k}：{hit_rate:.2%}")

        # 每种方法的 MRR 使用同一用例数作分母，保证横向可比。
        mrr = (
            reciprocal_rank_totals[method]
            / case_count
        )
        print(f"MRR@5：{mrr:.4f}")
        print(
            f"平均耗时："
            f"{mean(latency_ms[method]):.2f} ms"
        )
        print(
            f"中位耗时："
            f"{median(latency_ms[method]):.2f} ms"
        )

    print()
    print(
        f"RRF纯融合平均耗时："
        f"{mean(fusion_latency_ms):.4f} ms"
    )

    if args.rewrite_query:
        # 查询改写成本单独报告，不混入三种检索方法的延迟数字。
        print()
        print("========== 查询改写开销 ==========")
        print(
            f"实际调用模型次数：{rewrite_call_count}"
        )
        print(
            f"改写输入Token：{rewrite_input_tokens}"
        )
        print(
            f"改写输出Token：{rewrite_output_tokens}"
        )

        if rewrite_latency_ms:
            print(
                f"模型改写平均耗时："
                f"{mean(rewrite_latency_ms):.2f} ms"
            )


if __name__ == "__main__":
    main()
