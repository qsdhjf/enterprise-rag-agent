"""评估企业 Agent 是否能安全地区分“证据足够回答”和“应当拒答”的问题。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
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
from enterprise_agent.evaluation.retrieval import load_cases


# supported 与 partially_supported 均允许进入回答阶段，其余判断应拒答。
ALLOW_DECISIONS = {
    "supported",
    "partially_supported",
}


@dataclass(frozen=True, slots=True)
class AnswerabilityCase:
    """一条带可回答标签及正例相关文档集合的评测用例。"""

    case_id: str
    query: str
    source: str | None
    answerable: bool
    relevant_document_ids: frozenset[str]


def load_answerability_cases(
    positive_path: Path,
    negative_path: Path,
) -> list[AnswerabilityCase]:
    """合并正例检索集与人工负例集，并严格校验负例和用例 ID。

    正例继承已有检索标准文档；负例必须明确标记 ``answerable=false``，且标准
    相关文档列表必须为空。最终拒绝重复 case_id，避免重复样本扭曲统计。
    """

    cases: list[AnswerabilityCase] = []

    # 原有检索题都带标准相关文档，因此作为可回答正例复用。
    for item in load_cases(positive_path):
        cases.append(
            AnswerabilityCase(
                case_id=item.case_id,
                query=item.query,
                source=item.source,
                answerable=True,
                relevant_document_ids=(
                    item.relevant_document_ids
                ),
            )
        )

    # 负例必须经过人工核验，确保索引中确实没有足以回答的事实。
    with negative_path.open(
        "r",
        encoding="utf-8",
    ) as file:
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
                    f"负例文件第{line_number}行不是合法JSON"
                ) from error

            # 使用 ``is not False``，避免 0、空值等被宽松当作合法布尔标签。
            if record.get("answerable") is not False:
                raise ValueError(
                    f"负例第{line_number}行的"
                    "answerable必须为false"
                )

            relevant_ids = record.get(
                "relevant_document_ids"
            )

            if relevant_ids != []:
                raise ValueError(
                    f"负例第{line_number}行不能有"
                    "标准相关文档"
                )

            case_id = str(
                record.get("case_id") or ""
            ).strip()
            query = str(
                record.get("query") or ""
            ).strip()

            if not case_id or not query:
                raise ValueError(
                    f"负例第{line_number}行缺少"
                    "case_id或query"
                )

            cases.append(
                AnswerabilityCase(
                    case_id=case_id,
                    query=query,
                    source=record.get("source"),
                    answerable=False,
                    relevant_document_ids=frozenset(),
                )
            )

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("评测用例case_id存在重复")

    return cases


def elapsed_ms(started_at: float) -> float:
    """返回从高精度计时起点到当前时刻的毫秒数。"""

    return (perf_counter() - started_at) * 1000


def average(values: list[float]) -> float:
    """计算平均值；空列表安全地返回 0，便于技术失败场景生成报告。"""

    return mean(values) if values else 0.0


def middle(values: list[float]) -> float:
    """计算中位数；空列表安全地返回 0。"""

    return median(values) if values else 0.0


def first_relevant_rank(
    case: AnswerabilityCase,
    evidence,
) -> int | None:
    """对正例查找首条标准文档证据的排名；负例固定返回 ``None``。"""

    if not case.answerable:
        return None

    for rank, item in enumerate(
        evidence,
        start=1,
    ):
        if (
            item.document_id
            in case.relevant_document_ids
        ):
            return rank

    return None


def main() -> None:
    """运行查询改写、混合检索和证据审核，并汇总安全性、质量及性能指标。"""

    parser = argparse.ArgumentParser(
        description="评测企业Agent的证据充分性审核能力"
    )
    parser.add_argument(
        "--positive-cases",
        type=Path,
        required=True,
        help="有答案的正例JSONL",
    )
    parser.add_argument(
        "--negative-cases",
        type=Path,
        required=True,
        help="无答案的负例JSONL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="交给证据审核器的切块数量",
    )
    parser.add_argument(
        "--expected-index-count",
        type=int,
        default=780,
        help="混合索引预期切块数",
    )
    args = parser.parse_args()

    if not 1 <= args.top_k <= 10:
        raise ValueError("--top-k必须在1到10之间")

    cases = load_answerability_cases(
        positive_path=args.positive_cases.resolve(),
        negative_path=args.negative_cases.resolve(),
    )

    # bool 可作为 0/1 求和，得到正例数；评测要求正负样本都非空。
    positive_count = sum(
        case.answerable
        for case in cases
    )
    negative_count = len(cases) - positive_count

    if positive_count == 0 or negative_count == 0:
        raise ValueError("评测集必须同时包含正例和负例")

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

    # 混淆矩阵以“允许回答”为正类：TP=正确放行，FP=危险放行。
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0
    # 技术错误与语义误判分开计数，避免服务失败被错误算成安全拒答。
    technical_errors = 0
    strict_supported_count = 0
    strict_unsupported_count = 0
    end_to_end_success = 0

    positive_retrieval_hits = 0
    positive_reciprocal_rank_total = 0.0

    decision_counts: Counter[str] = Counter()

    rewrite_calls = 0
    rewrite_input_tokens = 0
    rewrite_output_tokens = 0
    judge_calls = 0
    judge_input_tokens = 0
    judge_output_tokens = 0

    retrieval_times: list[float] = []
    judge_times: list[float] = []
    total_times: list[float] = []

    try:
        if not opensearch_client.ping():
            raise RuntimeError("无法连接OpenSearch")

        actual_count = opensearch_client.count(
            index=(
                opensearch_settings.hybrid_index_name
            )
        )["count"]

        print(
            "评测索引："
            f"{opensearch_settings.hybrid_index_name}"
        )
        print(f"索引切块数：{actual_count}")
        print(
            f"评测用例：{len(cases)}，"
            f"正例 {positive_count}，"
            f"负例 {negative_count}"
        )

        if actual_count != args.expected_index_count:
            raise RuntimeError(
                "索引切块数不符合预期："
                f"期望{args.expected_index_count}，"
                f"实际{actual_count}"
            )

        # 预热本地 Embedding 模型和搜索连接，不计入正式耗时。
        print("正在预热BGE-M3和OpenSearch……")

        warmup_case = cases[0]

        hybrid_search_chunks(
            client=opensearch_client,
            index_name=(
                opensearch_settings.hybrid_index_name
            ),
            embedding_client=embedding_client,
            lexical_query=warmup_case.query,
            semantic_query=warmup_case.query,
            top_k=1,
            source=warmup_case.source,
        )

        print("预热完成，开始18题评测。")

        for case in cases:
            total_started = perf_counter()
            lexical_query = case.query
            rewrite_failed = False

            # 查询改写是可降级步骤；失败后仍用原问题继续检索。
            try:
                rewrite_result = rewrite_query(
                    query=case.query,
                    client=llm_client,
                )
                lexical_query = (
                    rewrite_result.retrieval_query
                )

                if rewrite_result.rewritten:
                    rewrite_calls += 1
                    rewrite_input_tokens += (
                        rewrite_result.input_tokens or 0
                    )
                    rewrite_output_tokens += (
                        rewrite_result.output_tokens or 0
                    )
            except Exception as error:
                rewrite_failed = True
                print(
                    f"\n用例 {case.case_id} "
                    f"查询改写失败，使用原问题：{error}"
                )

            retrieval_started = perf_counter()

            # 检索属于核心步骤，失败时该用例记技术错误并跳过语义判断。
            try:
                hybrid_results = hybrid_search_chunks(
                    client=opensearch_client,
                    index_name=(
                        opensearch_settings.hybrid_index_name
                    ),
                    embedding_client=embedding_client,
                    lexical_query=lexical_query,
                    semantic_query=case.query,
                    top_k=args.top_k,
                    source=case.source,
                )
            except Exception as error:
                technical_errors += 1
                decision_counts["retrieval_error"] += 1
                total_times.append(
                    elapsed_ms(total_started)
                )

                print()
                print(f"用例：{case.case_id}")
                print(f"预期：{'可回答' if case.answerable else '不可回答'}")
                print(f"结果：检索技术失败：{error}")
                continue

            retrieval_time = elapsed_ms(
                retrieval_started
            )
            retrieval_times.append(retrieval_time)

            # 审核器只需要融合命中的业务字段，不需要 RRF 排名元数据。
            evidence = [
                result.hit
                for result in hybrid_results
            ]

            relevant_rank = first_relevant_rank(
                case=case,
                evidence=evidence,
            )

            # 正例额外统计标准文档是否真的进入审核上下文及其倒数排名。
            if case.answerable:
                if relevant_rank is not None:
                    positive_retrieval_hits += 1
                    positive_reciprocal_rank_total += (
                        1.0 / relevant_rank
                    )

            judge_started = perf_counter()

            judge_result = judge_evidence(
                question=case.query,
                evidence=evidence,
                client=llm_client,
            )

            judge_time = elapsed_ms(judge_started)
            judge_times.append(judge_time)
            total_times.append(
                elapsed_ms(total_started)
            )

            # 无证据时审核器不会调用模型，因此令牌/调用计数按 used_model 累加。
            if judge_result.used_model:
                judge_calls += 1
                judge_input_tokens += (
                    judge_result.input_tokens or 0
                )
                judge_output_tokens += (
                    judge_result.output_tokens or 0
                )

            print()
            print(f"用例：{case.case_id}")
            print(
                f"预期："
                f"{'可回答' if case.answerable else '不可回答'}"
            )
            print(
                "标准文档在证据中的排名："
                + (
                    str(relevant_rank)
                    if relevant_rank is not None
                    else (
                        "未进入前"
                        f"{args.top_k}"
                        if case.answerable
                        else "负例不适用"
                    )
                )
            )
            print(
                f"审核判断：{judge_result.decision}"
            )
            print(
                "审核技术成功："
                f"{'是' if judge_result.judge_succeeded else '否'}"
            )

            if rewrite_failed:
                print("查询改写：失败后已回退")

            print(
                "事实状态："
                + ", ".join(
                    f"{fact.fact}={fact.status}"
                    for fact in judge_result.facts
                )
            )
            print(f"原因：{judge_result.reason}")
            print(
                f"耗时：检索 {retrieval_time:.2f} ms，"
                f"审核 {judge_time:.2f} ms"
            )

            # 技术失败不能算作正确拒答，否则“系统坏掉”也会虚增安全指标。
            if not judge_result.judge_succeeded:
                technical_errors += 1
                decision_counts["judge_error"] += 1
                print(
                    "评测结果：技术失败，"
                    "不计为语义判断正确"
                )
                continue

            decision_counts[
                judge_result.decision
            ] += 1

            # 将四类审核判断压缩成业务上的二元动作：允许回答或拒绝回答。
            allowed = (
                judge_result.decision
                in ALLOW_DECISIONS
            )

            if case.answerable:
                if allowed:
                    true_positive += 1
                    print("评测结果：正确允许回答")
                else:
                    false_negative += 1
                    print("评测结果：误拒答")

                # 端到端成功还要求标准证据已被检索到，不能只看审核器放行。
                if (
                    judge_result.decision
                    == "supported"
                ):
                    strict_supported_count += 1

                if (
                    allowed
                    and relevant_rank is not None
                ):
                    end_to_end_success += 1
            else:
                if allowed:
                    false_positive += 1
                    print("评测结果：危险放行")
                else:
                    true_negative += 1
                    end_to_end_success += 1
                    print("评测结果：正确拒答")

                if (
                    judge_result.decision
                    == "unsupported"
                ):
                    strict_unsupported_count += 1

    finally:
        # 确保评测中途异常时也释放两个外部服务客户端。
        embedding_client.close()
        opensearch_client.close()

    total_count = len(cases)

    # 分别计算正例召回与负例拒答，再取平均得到不受类别比例影响的平衡准确率。
    positive_allow_rate = (
        true_positive / positive_count
    )
    negative_rejection_rate = (
        true_negative / negative_count
    )
    false_rejection_rate = (
        false_negative / positive_count
    )
    false_answer_rate = (
        false_positive / negative_count
    )
    balanced_accuracy = (
        positive_allow_rate
        + negative_rejection_rate
    ) / 2
    overall_accuracy = (
        true_positive + true_negative
    ) / total_count
    # 技术成功率单独体现服务/格式稳定性，不与语义准确率混为一谈。
    technical_success_rate = (
        total_count - technical_errors
    ) / total_count

    print()
    print("========== 可回答性评测汇总 ==========")
    print(f"总用例：{total_count}")
    print(f"正例：{positive_count}")
    print(f"负例：{negative_count}")
    print()

    print(
        f"正例允许回答率：{positive_allow_rate:.2%}"
    )
    print(
        "正例严格supported率："
        f"{strict_supported_count / positive_count:.2%}"
    )
    print(
        f"误拒答率：{false_rejection_rate:.2%}"
    )
    print()

    print(
        "负例安全拒答率："
        f"{negative_rejection_rate:.2%}"
    )
    print(
        "负例严格unsupported率："
        f"{strict_unsupported_count / negative_count:.2%}"
    )
    print(
        f"危险放行率：{false_answer_rate:.2%}"
    )
    print()

    print(
        f"平衡准确率：{balanced_accuracy:.2%}"
    )
    print(
        f"总体准确率：{overall_accuracy:.2%}"
    )
    print(
        f"审核技术成功率：{technical_success_rate:.2%}"
    )
    print(
        "端到端安全成功率："
        f"{end_to_end_success / total_count:.2%}"
    )

    print()
    print("========== 正例检索质量 ==========")
    print(
        f"证据Hit@{args.top_k}："
        f"{positive_retrieval_hits / positive_count:.2%}"
    )
    print(
        f"证据MRR@{args.top_k}："
        f"{positive_reciprocal_rank_total / positive_count:.4f}"
    )

    print()
    print("========== 判断分布 ==========")

    for decision, count in sorted(
        decision_counts.items()
    ):
        print(f"{decision}：{count}")

    print()
    print("========== 性能与Token ==========")
    print(
        f"平均检索耗时："
        f"{average(retrieval_times):.2f} ms"
    )
    print(
        f"检索中位耗时："
        f"{middle(retrieval_times):.2f} ms"
    )
    print(
        f"平均审核耗时："
        f"{average(judge_times):.2f} ms"
    )
    print(
        f"审核中位耗时："
        f"{middle(judge_times):.2f} ms"
    )
    print(f"查询改写模型调用：{rewrite_calls}")
    print(
        f"查询改写Token：输入{rewrite_input_tokens}，"
        f"输出{rewrite_output_tokens}"
    )
    print(f"证据审核模型调用：{judge_calls}")
    print(
        f"证据审核Token：输入{judge_input_tokens}，"
        f"输出{judge_output_tokens}"
    )


if __name__ == "__main__":
    main()
