"""命令行评估 OpenSearch BM25 检索的 Hit@K 与 MRR@5。"""

from __future__ import annotations
from enterprise_agent.infrastructure.config import load_llm_settings
from enterprise_agent.infrastructure.llm_client import OpenAICompatibleLLMClient
from enterprise_agent.retrieval.query_rewriter import rewrite_query
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from enterprise_agent.infrastructure.opensearch_client import (
    create_opensearch_client,
    load_opensearch_settings,
)
from enterprise_agent.retrieval.keyword_search import search_chunks


@dataclass(frozen=True)
class RetrievalCase:
    """一条人工标注的检索问题及其可接受的相关文档 ID 集合。"""

    case_id: str
    query: str
    source: str | None
    relevant_document_ids: frozenset[str]


def load_cases(path: Path) -> list[RetrievalCase]:
    """从 JSONL 加载检索评测集，并校验 JSON 与标准答案非空。

    空行被忽略；解析错误会附带原始行号，便于修复标注文件。
    """

    cases: list[RetrievalCase] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第 {line_number} 行不是合法 JSON"
                ) from error

            # frozenset 去除重复 ID，并支持快速的 O(1) 相关性判断。
            relevant_ids = frozenset(
                str(document_id)
                for document_id in record["relevant_document_ids"]
            )

            if not relevant_ids:
                raise ValueError(
                    f"用例 {record['case_id']} 没有标准答案"
                )

            cases.append(
                RetrievalCase(
                    case_id=str(record["case_id"]),
                    query=str(record["query"]),
                    source=record.get("source"),
                    relevant_document_ids=relevant_ids,
                )
            )

    if not cases:
        raise ValueError("评测文件中没有有效用例")

    return cases


def main() -> None:
    """解析命令行参数，逐用例检索，并打印明细与总体指标。"""

    parser = argparse.ArgumentParser(
        description="评估 OpenSearch 的文档召回效果",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSONL 格式的人工标注评测集",
    )
    parser.add_argument(
    "--rewrite-query",
    action="store_true",
    help="对包含中文的查询生成英文检索词",
)
    
    args = parser.parse_args()

    cases = load_cases(args.cases)

    settings = load_opensearch_settings()
    client = create_opensearch_client(settings)
    rewrite_client = None

    if args.rewrite_query:
        rewrite_client = OpenAICompatibleLLMClient(
        load_llm_settings()
    )

    # Hit@K 统计前 K 个结果中是否至少出现一个相关文档。
    ks = (1, 3, 5)
    hit_counts = {k: 0 for k in ks}
    reciprocal_rank_total = 0.0

    try:
        if not client.ping():
            raise RuntimeError("无法连接 OpenSearch")

        for case in cases:
            retrieval_query = case.query
            rewrite_result = None

            # 可选查询改写只影响 BM25 实际检索词，评测标签仍对应原问题。
            if rewrite_client is not None:
                rewrite_result = rewrite_query(
                    query=case.query,
                    client=rewrite_client,
                )
                retrieval_query = rewrite_result.retrieval_query
            results = search_chunks(
                client=client,
                index_name=settings.index_name,
                query=retrieval_query,
                top_k=5,
                source=case.source,
                max_chunks_per_document=1,
            )

            # 检索阶段限制每篇文档一个块，所以这里直接以文档 ID 评测排名。
            returned_ids = [
                result.document_id
                for result in results
            ]

            first_relevant_rank: int | None = None

            # MRR 只关心第一篇相关文档出现的位置。
            for rank, document_id in enumerate(
                returned_ids,
                start=1,
            ):
                if document_id in case.relevant_document_ids:
                    first_relevant_rank = rank
                    break

            for k in ks:
                top_k_ids = returned_ids[:k]

                if any(
                    document_id in case.relevant_document_ids
                    for document_id in top_k_ids
                ):
                    hit_counts[k] += 1

            # 前五名没有命中时该用例的倒数排名记为 0。
            reciprocal_rank = (
                1.0 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            )
            reciprocal_rank_total += reciprocal_rank

            print()
            print(f"用例：{case.case_id}")
            print(f"查询：{case.query}")
            print(f"限定来源：{case.source or '全部'}")
            print(
                "标准文档："
                + ", ".join(case.relevant_document_ids)
            )
            print(
                "返回文档："
                + (
                    ", ".join(returned_ids)
                    if returned_ids
                    else "无"
                )
            )
            print(
                "首个正确排名："
                + (
                    str(first_relevant_rank)
                    if first_relevant_rank is not None
                    else "前5名未命中"
                )
            )
            print(f"实际检索词：{retrieval_query}")

            if rewrite_result is not None:
                print(
                    f"是否经过模型改写："
                    f"{'是' if rewrite_result.rewritten else '否'}"
                )

    finally:
        # 即使评测中途报错，也释放 OpenSearch 连接。
        client.close()

    case_count = len(cases)

    print()
    print("========== 总体指标 ==========")
    print(f"用例数量：{case_count}")

    for k in ks:
        hit_rate = hit_counts[k] / case_count
        print(f"Hit@{k}：{hit_rate:.2%}")

    # MRR 是每个用例“首个正确排名倒数”的算术平均。
    mrr = reciprocal_rank_total / case_count
    print(f"MRR@5：{mrr:.4f}")


if __name__ == "__main__":
    main()
