"""使用 OpenSearch BM25 关键词检索企业文档切块。"""

from __future__ import annotations

import argparse

from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.retrieval.keyword_search import search_chunks


def main() -> None:
    """读取检索参数，执行关键词搜索并输出便于人工查看的摘要。"""
    parser = argparse.ArgumentParser(
        description="从 OpenSearch 检索企业文档切块",
    )

    # --query 是必填检索文本。
    parser.add_argument(
        "--query",
        required=True,
        help="需要搜索的问题或关键词",
    )

    # --top-k 限制最多返回多少条结果，默认取前 5 条。
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最多返回多少个切块",
    )

    # --source 可选；传入后只检索指定的数据源。
    parser.add_argument(
        "--source",
        help="只搜索指定数据源，例如 jira 或 slack",
    )

    args = parser.parse_args()

    # CLI 自己管理客户端生命周期，避免命令结束后遗留连接。
    settings = load_opensearch_settings()
    client = create_opensearch_client(settings)

    try:
        if not client.ping():
            raise RuntimeError("无法连接 OpenSearch")

        # search_chunks 是 CLI 与关键词检索实现之间的适配边界。
        results = search_chunks(
            client=client,
            index_name=settings.index_name,
            query=args.query,
            top_k=args.top_k,
            source=args.source,
        )
    finally:
        client.close()

    # 空结果属于正常情况，不作为命令执行错误处理。
    if not results:
        print("没有找到相关切块")
        return

    # 只展示正文前 400 个字符，避免终端被长文档淹没。
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())[:400]

        print()
        print(f"第 {rank} 名")
        print(f"得分：{result.score:.4f}")
        print(f"标题：{result.title}")
        print(f"来源：{result.source}")
        print(f"切块：{result.chunk_id}")
        print(f"原文档：{result.document_id}")
        print(f"原文序号：{result.sequence}")
        print(f"内容预览：{preview}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
