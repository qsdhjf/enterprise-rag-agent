"""通过统一企业搜索工具运行查询，并展示工具级诊断信息。"""

from argparse import ArgumentParser
import sys

from enterprise_agent.infrastructure.config import (
    load_embedding_settings,
    load_llm_settings,
    load_reranker_settings,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.retrieval.enterprise_search import (
    EnterpriseSearchInput,
    EnterpriseSearchTool,
)
from enterprise_agent.infrastructure.llm_client import (
    OpenAICompatibleLLMClient,
)
from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.infrastructure.reranker_client import (
    TEIRerankerClient,
)


def main() -> None:
    """组装企业搜索所需依赖，执行一次查询并打印融合结果。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = ArgumentParser(
        description="测试企业知识搜索工具",
    )
    # --query 是原始用户问题，工具内部可据此生成词法查询。
    parser.add_argument(
        "--query",
        required=True,
        help="需要检索的问题或关键词",
    )
    # --source 可限制数据源；--top-k 控制最终证据数。
    parser.add_argument(
        "--source",
        help="限定数据源，例如jira或slack",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回的证据切块数量",
    )
    args = parser.parse_args()

    # 企业搜索同时依赖查询改写模型、向量模型和 OpenSearch。
    llm_client = OpenAICompatibleLLMClient(
        load_llm_settings(),
    )
    embedding_client = (
        OpenAICompatibleEmbeddingClient(
            load_embedding_settings(),
        )
    )

    reranker_client = TEIRerankerClient(
        load_reranker_settings(),
    )

    opensearch_settings = (
        load_opensearch_settings()
    )
    opensearch_client = (
        create_opensearch_client(
            opensearch_settings,
        )
    )

    # EnterpriseSearchTool 是 CLI 与完整检索流水线之间的统一适配层。
    tool = EnterpriseSearchTool(
        llm_client=llm_client,
        embedding_client=embedding_client,
        opensearch_client=opensearch_client,
        index_name=(
            opensearch_settings.hybrid_index_name
        ),
        reranker_client=reranker_client,
    )

    try:
        # 输入模型把命令行原始值转换为工具约定的结构化请求。
        result = tool.run(
            EnterpriseSearchInput(
                query=args.query,
                source=args.source,
                top_k=args.top_k,
            )
        )
    finally:
        # 按与创建相反的方向释放所有外部客户端。
        opensearch_client.close()
        reranker_client.close()
        embedding_client.close()
        llm_client.close()

    print("\n========== 工具调用 ==========")
    print(f"工具名称：{tool.name}")
    print(f"原始搜索词：{result.query}")
    print(f"BM25搜索词：{result.lexical_query}")
    print(f"限定来源：{result.source or '全部'}")
    print(f"改写成功：{result.rewrite_succeeded}")
    print(f"返回切块：{len(result.hits)}")

    print("\n========== 工具观察结果 ==========")

    # 每条结果同时展示 RRF 分数及两路原始排名，便于调试融合效果。
    for number, result_item in enumerate(
        result.hits,
        start=1,
    ):
        hit = result_item.hit

        print(f"\n第 {number} 条证据")
        print(f"标题：{hit.title}")
        print(f"来源：{hit.source}")
        print(f"文档：{hit.document_id}")
        print(f"切块：{hit.chunk_id}")
        print(f"原文序号：{hit.sequence}")
        print(
            f"RRF得分："
            f"{result_item.rrf_score:.6f}"
        )
        print(
            f"Reranker rank: {result_item.reranker_rank}"
        )
        reranker_score_text = (
            f"{result_item.reranker_score:.6f}"
            if result_item.reranker_score is not None
            else "not run"
        )
        print(f"Reranker score: {reranker_score_text}")
        print(
            f"BM25排名："
            f"{result_item.bm25_rank}"
        )
        print(
            f"向量排名："
            f"{result_item.vector_rank}"
        )
        print(f"内容预览：{hit.text[:400]}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
