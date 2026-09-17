"""无需 OpenSearch 的轻量 JSONL 关键词检索，适合本地验证和小数据集。"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchResult:
    """本地关键词匹配的得分和文本块元数据。"""

    score: int
    chunk_id: str
    document_id: str
    source: str
    title: str
    sequence: int
    text: str
    created_at: str | None
    original_location: str | None


def _token_counts(text: str) -> Counter[str]:
    """将文本统一为小写，并统计 ASCII 字母、数字和下划线词项频次。"""

    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    return Counter(tokens)


def search_jsonl(
    index_path: Path,
    query: str,
    top_k: int = 5,
    max_chunks_per_document: int = 2,
) -> list[SearchResult]:
    """扫描 JSONL 中的全部块，以词频加权打分后返回最相关结果。

    这是简单的精确词项匹配：标题中的一次命中计 3 分，正文命中计 1 分，
    再乘以查询词自身出现次数。它不进行分词语言学处理或语义匹配。
    """

    if not index_path.is_file():
        raise FileNotFoundError(f"切块文件不存在：{index_path}")

    if top_k < 1:
        raise ValueError("top_k 必须大于 0")

    if max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document 必须大于 0")

    query_counts = _token_counts(query)

    if not query_counts:
        raise ValueError("查询中没有可检索的关键词")

    candidates: list[SearchResult] = []

    with index_path.open("r", encoding="utf-8") as index_file:
        for line in index_file:
            if not line.strip():
                continue

            record = json.loads(line)

            title = str(record.get("title") or "")
            text = str(record.get("text") or "")

            title_counts = _token_counts(title)
            text_counts = _token_counts(text)

            # 标题通常是更强的主题信号，因此其词频乘以 3。
            score = sum(
                query_count
                * (
                    title_counts.get(term, 0) * 3
                    + text_counts.get(term, 0)
                )
                for term, query_count in query_counts.items()
            )

            if score <= 0:
                continue

            candidates.append(
                SearchResult(
                    score=score,
                    chunk_id=str(record["chunk_id"]),
                    document_id=str(record["document_id"]),
                    source=str(record["source"]),
                    title=title,
                    sequence=int(record["sequence"]),
                    text=text,
                    created_at=record.get("created_at"),
                    original_location=record.get("original_location"),
                )
            )

    # 先全局按分数排序，再做文档级配额，保留每篇文档最强的块。
    candidates.sort(key=lambda result: result.score, reverse=True)

    selected: list[SearchResult] = []
    document_counts: Counter[str] = Counter()

    for candidate in candidates:
        # 控制结果多样性，避免长文档因重复词多而霸占列表。
        if document_counts[candidate.document_id] >= max_chunks_per_document:
            continue

        selected.append(candidate)
        document_counts[candidate.document_id] += 1

        if len(selected) >= top_k:
            break

    return selected
