"""检索候选证据并单独测试“证据是否足以回答问题”的命令行工具。"""

from __future__ import annotations

from argparse import ArgumentParser

from enterprise_agent.infrastructure.config import (
    load_embedding_settings,
    load_llm_settings,
)
from enterprise_agent.infrastructure.embedding_client import (
    OpenAICompatibleEmbeddingClient,
)
from enterprise_agent.generation.evidence_judge import judge_evidence
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


# 将审核器内部枚举值翻译成便于人工阅读的中文标签。
STATUS_LABELS = {
    "supported": "有证据支持",
    "missing": "证据缺失",
    "conflicting": "证据冲突",
}

DECISION_LABELS = {
    "supported": "可以完整回答",
    "partially_supported": "只能部分回答",
    "unsupported": "不能回答",
    "conflicting": "证据存在冲突",
}


def main() -> None:
    """执行查询改写、混合检索和证据审核，并打印逐事实结论。"""
    parser = ArgumentParser(
        description="测试企业证据是否足以回答问题"
    )
    # --question 是审核目标；--source 可缩小检索范围。
    parser.add_argument(
        "--question",
        required=True,
        help="需要审核的问题",
    )
    parser.add_argument(
        "--source",
        help="限定数据源，例如jira或slack",
    )
    # --top-k 控制送给审核模型的证据数量，而非最终答案数量。
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="交给审核器的证据切块数量",
    )
    args = parser.parse_args()

    # 同一个 LLM 客户端先用于改写查询，后用于证据审核。
    llm_client = OpenAICompatibleLLMClient(
        load_llm_settings()
    )

    # 为 BM25 准备更偏关键词的检索词；失败时仍可用原问题继续。
    try:
        rewrite_result = rewrite_query(
            query=args.question,
            client=llm_client,
        )
        lexical_query = rewrite_result.retrieval_query
    except Exception as error:
        print(
            f"查询改写失败，使用原始问题：{error}"
        )
        lexical_query = args.question

    # 混合检索还需要 Embedding 客户端和 OpenSearch 客户端。
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

        # BM25 使用 lexical_query，向量检索保留原始问题的完整语义。
        hybrid_results = hybrid_search_chunks(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            embedding_client=embedding_client,
            lexical_query=lexical_query,
            semantic_query=args.question,
            top_k=args.top_k,
            source=args.source,
        )
    finally:
        # 检索阶段结束后立即释放不再需要的外部连接。
        embedding_client.close()
        opensearch_client.close()

    # 审核器只需要融合结果中的原始命中切块，不需要排名包装对象。
    evidence = [
        result.hit
        for result in hybrid_results
    ]

    # 此工具只审核证据，不生成最终答案，便于隔离验证审核逻辑。
    judge_result = judge_evidence(
        question=args.question,
        evidence=evidence,
        client=llm_client,
    )

    print("\n========== 检索信息 ==========")
    print(f"用户问题：{args.question}")
    print(f"BM25检索词：{lexical_query}")
    print(f"证据数量：{len(evidence)}")

    for number, item in enumerate(
        evidence,
        start=1,
    ):
        print(
            f"E{number}：{item.source} | "
            f"{item.title} | {item.chunk_id}"
        )

    print("\n========== 证据审核结果 ==========")
    print(
        "最终判断："
        f"{judge_result.decision} "
        f"({DECISION_LABELS[judge_result.decision]})"
    )
    print(
        "审核器是否正常完成："
        f"{'是' if judge_result.judge_succeeded else '否'}"
    )

    if judge_result.error_code:
        print(f"错误代码：{judge_result.error_code}")

    print(f"总体原因：{judge_result.reason}")

    print("\n========== 必需事实检查 ==========")

    # 按审核器拆出的必需事实逐项展示支持、缺失或冲突状态。
    for number, fact in enumerate(
        judge_result.facts,
        start=1,
    ):
        print()
        print(f"事实 {number}：{fact.fact}")
        print(
            f"状态：{fact.status} "
            f"({STATUS_LABELS[fact.status]})"
        )
        print(
            "结论："
            f"{fact.conclusion or '无'}"
        )
        print(
            "支持证据："
            + (
                ", ".join(fact.evidence_ids)
                if fact.evidence_ids
                else "无"
            )
        )

    print("\n========== 审核调用信息 ==========")
    print(f"模型：{judge_result.model or '未调用'}")
    print(f"输入Token：{judge_result.input_tokens}")
    print(f"输出Token：{judge_result.output_tokens}")
    print(f"请求ID：{judge_result.request_id}")


if __name__ == "__main__":
    # 直接运行模块时启动 CLI。
    main()
