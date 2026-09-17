"""在本地 JSONL 切块文件中搜索，并可预览最终 RAG 提示词。"""

from argparse import ArgumentParser
from pathlib import Path

from enterprise_agent.retrieval.local_search import search_jsonl
from enterprise_agent.generation.prompt_builder import build_rag_prompt


def main() -> None:
    """解析本地索引与查询参数，展示命中结果或组装后的提示词。"""
    parser = ArgumentParser(description="本地企业文档检索")

    # --index 指向无需 OpenSearch 即可读取的 JSONL 切块索引。
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="JSONL 切块文件路径",
    )
    # --query 是要匹配的用户问题或关键词。
    parser.add_argument(
        "--query",
        required=True,
        help="要检索的问题或关键词",
    )
    # --top-k 限制返回数量，避免一次输出过多正文。
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最多返回多少个相关切块",
    )
    # --show-prompt 用于调试“证据如何进入模型上下文”。
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="显示准备交给大模型的完整证据包",
    )

    args = parser.parse_args()

    # 本地搜索适合在没有外部服务时验证切块和排序效果。
    results = search_jsonl(
        index_path=args.index,
        query=args.query,
        top_k=args.top_k,
    )

    print(f"查询：{args.query}")
    print(f"命中结果：{len(results)}")

    for rank, result in enumerate(results, start=1):
        preview = result.text.replace("\n", " ")[:300]

        print(f"\n第 {rank} 名")
        print(f"得分：{result.score}")
        print(f"标题：{result.title}")
        print(f"来源：{result.source}")
        print(f"切块：{result.chunk_id}")
        print(f"原文档：{result.document_id}")
        print(f"原文序号：{result.sequence}")
        print(f"内容预览：{preview}")

    if args.show_prompt:
        # 提示词构建器把用户问题和命中切块整理成系统/用户两段消息。
        prompt = build_rag_prompt(
            question=args.query,
            results=results,
        )

        print("\n========== SYSTEM PROMPT ==========")
        print(prompt.system_prompt)
        print("\n========== USER PROMPT ==========")
        print(prompt.user_prompt)
        print(f"\n证据数量：{prompt.evidence_count}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
