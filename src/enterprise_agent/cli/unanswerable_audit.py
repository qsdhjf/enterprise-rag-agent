"""批量检查候选“不可回答”问题的最相关检索证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    hybrid_search_chunks,
)
from enterprise_agent.retrieval.query_rewriter import rewrite_query


def load_candidates(path: Path) -> list[dict]:
    """从 JSONL 读取并校验候选用例，忽略纯空白行。"""
    candidates: list[dict] = []

    # 逐行解析可以准确报告坏数据所在的行号。
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第{line_number}行不是合法JSON"
                ) from error

            # case_id 用于报告定位，query 是真正执行检索的文本。
            case_id = str(
                record.get("case_id") or ""
            ).strip()
            query = str(
                record.get("query") or ""
            ).strip()

            if not case_id or not query:
                raise ValueError(
                    f"第{line_number}行缺少case_id或query"
                )

            # source 保持可选；空值表示允许从全部数据源召回。
            candidates.append(
                {
                    "case_id": case_id,
                    "query": query,
                    "source": record.get("source"),
                }
            )

    if not candidates:
        raise ValueError("没有候选用例")

    return candidates


def optional_rank(rank: int | None) -> str:
    """格式化可能不存在的单路检索排名。"""
    if rank is None:
        return "未进入候选集"

    return str(rank)


def optional_score(score: float | None) -> str:
    """格式化可能不存在的单路检索分数。"""
    if score is None:
        return "无"

    return f"{score:.4f}"


def main() -> None:
    """逐个候选问题运行混合检索，输出供人工判定的证据。"""
    parser = argparse.ArgumentParser(
        description="审查候选无答案问题的最相关证据"
    )
    # --cases 指向每行至少含 case_id 和 query 的 JSONL 文件。
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="候选无答案JSONL文件",
    )
    # --top-k 控制每个用例展示的融合候选数量。
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每题展示多少个融合结果",
    )
    args = parser.parse_args()

    # 在创建模型和搜索客户端前先完成输入文件校验。
    candidates = load_candidates(
        args.cases.resolve()
    )

    # 一个批次复用同一组客户端，避免每个候选重复建立连接。
    llm_client = OpenAICompatibleLLMClient(
        load_llm_settings()
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        load_embedding_settings()
    )

    opensearch_settings = load_opensearch_settings()
    opensearch_client = create_opensearch_client(
        opensearch_settings
    )

    try:
        if not opensearch_client.ping():
            raise RuntimeError("无法连接OpenSearch")

        for candidate in candidates:
            query = candidate["query"]

            # 查询改写失败不会中止整批审计，而是回退到原问题。
            try:
                rewrite_result = rewrite_query(
                    query=query,
                    client=llm_client,
                )
                lexical_query = (
                    rewrite_result.retrieval_query
                )
            except Exception as error:
                print(
                    f"查询改写失败，使用原问题：{error}"
                )
                lexical_query = query

            # 每篇文档最多保留两个切块，减少单一文档垄断候选结果。
            results = hybrid_search_chunks(
                client=opensearch_client,
                index_name=(
                    opensearch_settings.hybrid_index_name
                ),
                embedding_client=embedding_client,
                lexical_query=lexical_query,
                semantic_query=query,
                top_k=args.top_k,
                source=candidate["source"],
                max_chunks_per_document=2,
            )

            print()
            print("=" * 70)
            print(f"用例：{candidate['case_id']}")
            print(f"问题：{query}")
            print(
                "限定来源："
                f"{candidate['source'] or '全部'}"
            )
            print(f"BM25检索词：{lexical_query}")

            if not results:
                print("没有检索结果")
                continue

            # 展示两路排名和较长正文预览，供人工判断是否真的不可回答。
            for rank, result in enumerate(
                results,
                start=1,
            ):
                hit = result.hit
                preview = " ".join(
                    hit.text.split()
                )[:800]

                print()
                print(f"候选证据 {rank}")
                print(
                    f"RRF得分：{result.rrf_score:.6f}"
                )
                print(
                    "BM25："
                    f"排名 {optional_rank(result.bm25_rank)}，"
                    f"分数 {optional_score(result.bm25_score)}"
                )
                print(
                    "向量："
                    f"排名 {optional_rank(result.vector_rank)}，"
                    f"分数 {optional_score(result.vector_score)}"
                )
                print(f"标题：{hit.title}")
                print(f"来源：{hit.source}")
                print(f"文档ID：{hit.document_id}")
                print(f"切块ID：{hit.chunk_id}")
                print(f"原文序号：{hit.sequence}")
                print(
                    "原始位置："
                    f"{hit.original_location or '未提供'}"
                )
                print(f"内容预览：{preview}")

            print()
            print(
                "人工判定要求：以上证据是否明确、直接、"
                "完整地支持问题所要求的全部事实？"
            )

    finally:
        # 无论某个候选是否失败，批次结束时都关闭共享客户端。
        embedding_client.close()
        opensearch_client.close()


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
