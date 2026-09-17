"""使用语义向量在 OpenSearch 中检索企业文档切块。"""

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
from enterprise_agent.retrieval.vector_search import (
    vector_search_chunks,
)


def main() -> None:
    """把用户问题编码为向量，检索近邻切块并打印结果。"""
    parser = argparse.ArgumentParser(
        description="使用语义向量检索企业文档切块"
    )
    # --query 会同时作为待编码的语义查询文本。
    parser.add_argument(
        "--query",
        required=True,
        help="需要搜索的问题",
    )
    # --top-k 控制最终展示的近邻数量。
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最多返回多少个切块",
    )
    # --source 用于把检索范围缩小到单一数据源。
    parser.add_argument(
        "--source",
        help="限定数据源，例如 jira 或 slack",
    )
    args = parser.parse_args()

    # 搜索服务配置与 Embedding 模型配置分开加载。
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
            raise RuntimeError("无法连接 OpenSearch")

        # 检索函数内部负责生成查询向量并提交向量近邻查询。
        results = vector_search_chunks(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            embedding_client=embedding_client,
            query=args.query,
            top_k=args.top_k,
            source=args.source,
        )
    finally:
        # 两个客户端均持有外部连接，必须成对释放。
        embedding_client.close()
        opensearch_client.close()

    # 没有命中不代表程序故障，给出提示后正常结束。
    if not results:
        print("没有找到相关切块")
        return

    print("\n========== 纯向量检索结果 ==========")

    for rank, result in enumerate(results, start=1):
        preview = " ".join(
            result.text.split()
        )[:400]

        print()
        print(f"第 {rank} 名")
        print(f"向量得分：{result.score:.4f}")
        print(f"标题：{result.title}")
        print(f"来源：{result.source}")
        print(f"切块：{result.chunk_id}")
        print(f"原文档：{result.document_id}")
        print(f"原文序号：{result.sequence}")
        print(f"内容预览：{preview}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
