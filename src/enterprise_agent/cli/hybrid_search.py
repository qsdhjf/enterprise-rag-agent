"""组合 BM25 与语义向量结果，演示基于 RRF 的混合检索。"""

from __future__ import annotations

import argparse

from enterprise_agent.infrastructure.config import (
    load_embedding_settings,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.retrieval.hybrid_search import (
    hybrid_search_chunks,
)
from enterprise_agent.infrastructure.config import load_llm_settings
from enterprise_agent.infrastructure.llm_client import OpenAICompatibleLLMClient
from enterprise_agent.retrieval.query_rewriter import rewrite_query

def format_rank(rank: int | None) -> str:
    """把可缺省的名次转换成适合终端展示的文字。"""
    if rank is None:
        return "未命中"

    return str(rank)


def format_score(score: float | None) -> str:
    """把可缺省的检索分数格式化为四位小数。"""
    if score is None:
        return "无"

    return f"{score:.4f}"


def main() -> None:
    """准备词法与语义查询，执行混合检索并显示两路排名。"""
    parser = argparse.ArgumentParser(
        description="使用BM25和向量检索进行RRF混合搜索"
    )
    # --query 始终保留用户原问题，并用于语义向量检索。
    parser.add_argument(
        "--query",
        required=True,
        help="用户的原始问题，同时作为向量检索问题",
    )
    # --lexical-query 可显式覆盖 BM25 查询，省略时尝试自动改写。
    parser.add_argument(
        "--lexical-query",
        help=(
            "提供给BM25的关键词查询；"
            "不指定时使用原始问题"
        ),
    )
    # --source 与 --top-k 分别控制来源过滤和最终返回数量。
    parser.add_argument(
        "--source",
        help="限定数据源，例如jira或slack",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最终返回多少个切块",
    )
    args = parser.parse_args()

    # 词法查询有两条路径：人工指定，或由模型根据原问题生成。
    if args.lexical_query:
        # 用户手动提供BM25查询时，不调用DeepSeek。
        lexical_query = args.lexical_query.strip()
        rewrite_mode = "手动指定"
    else:
        # 没有手动提供时，自动调用阶段一的查询改写器。
        llm_settings = load_llm_settings()
        llm_client = OpenAICompatibleLLMClient(
            llm_settings
        )

        try:
            # 改写只服务于关键词召回，不替换下面的语义查询文本。
            rewrite_result = rewrite_query(
                query=args.query,
                client=llm_client,
            )
            lexical_query = rewrite_result.retrieval_query

            rewrite_mode = (
                "模型自动改写"
                if rewrite_result.rewritten
                else "原问题无需改写"
            )
        except Exception as error:
            # 改写失败时，向量检索仍然可以继续工作。
            print(
                f"查询改写失败，回退到原问题：{error}"
            )
            lexical_query = args.query
            rewrite_mode = "改写失败，使用原问题"

    # 准备混合检索依赖的搜索服务与向量服务。
    opensearch_settings = load_opensearch_settings()
    embedding_settings = load_embedding_settings()

    opensearch_client = create_opensearch_client(
        opensearch_settings
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        embedding_settings
    )

    try:
        if not opensearch_client.ping():
            raise RuntimeError("无法连接OpenSearch")

        # 检索器分别取两路候选，再用 RRF 名次融合生成统一排序。
        results = hybrid_search_chunks(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            embedding_client=embedding_client,

            # BM25使用关键词查询。
            lexical_query=lexical_query,

            # 向量检索始终理解用户的原始问题。
            semantic_query=args.query,

            top_k=args.top_k,
            source=args.source,
        )
    finally:
        # 搜索结束或失败后都释放 OpenSearch 与 Embedding 客户端。
        embedding_client.close()
        opensearch_client.close()

    # 空命中是合法业务结果，提示后正常返回。
    if not results:
        print("没有找到相关切块")
        return

    print("\n========== RRF混合检索结果 ==========")
    print(f"原始问题：{args.query}")
    print(f"查询改写方式：{rewrite_mode}")
    print(f"BM25问题：{lexical_query}")

    for rank, result in enumerate(results, start=1):
        hit = result.hit
        preview = " ".join(hit.text.split())[:400]

        print()
        print(f"第 {rank} 名")
        print(f"RRF得分：{result.rrf_score:.6f}")
        print(
            "BM25："
            f"排名 {format_rank(result.bm25_rank)}，"
            f"原始分 {format_score(result.bm25_score)}"
        )
        print(
            "向量："
            f"排名 {format_rank(result.vector_rank)}，"
            f"原始分 {format_score(result.vector_score)}"
        )
        print(f"标题：{hit.title}")
        print(f"来源：{hit.source}")
        print(f"切块：{hit.chunk_id}")
        print(f"原文档：{hit.document_id}")
        print(f"原文序号：{hit.sequence}")
        print(f"内容预览：{preview}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
