"""演示如何把自然语言问题改写成适合关键词检索的查询。"""

from __future__ import annotations

import argparse

from enterprise_agent.infrastructure.config import load_llm_settings
from enterprise_agent.infrastructure.llm_client import OpenAICompatibleLLMClient
from enterprise_agent.retrieval.query_rewriter import rewrite_query


def main() -> None:
    """接收原始问题，调用查询改写器并打印改写及 Token 统计。"""
    parser = argparse.ArgumentParser(
        description="把用户问题改写成适合英文文档检索的查询",
    )
    # --query 保留用户原意，改写器只生成额外的检索表达式。
    parser.add_argument(
        "--query",
        required=True,
        help="用户原始问题",
    )
    args = parser.parse_args()

    # 改写器通过统一的 OpenAI 兼容客户端调用配置中的模型。
    settings = load_llm_settings()
    client = OpenAICompatibleLLMClient(settings)

    # 返回值同时包含原问题、检索问题和模型调用统计。
    result = rewrite_query(
        query=args.query,
        client=client,
    )

    print(f"是否改写：{'是' if result.rewritten else '否'}")
    print(f"原始问题：{result.original_query}")
    print(f"检索问题：{result.retrieval_query}")
    print(f"输入 Token：{result.input_tokens}")
    print(f"输出 Token：{result.output_tokens}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
