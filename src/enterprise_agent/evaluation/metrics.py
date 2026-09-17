"""Document-level, binary-relevance metrics with explicit no-gold semantics."""

import math


def retrieval_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    if not relevant_ids:
        return {"hit": None, "recall": None, "mrr": None, "ndcg": None,
                "no_result": float(not ranked)}
    hits = [int(item in relevant_ids) for item in ranked]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant_ids))))
    return {"hit": float(any(hits)), "recall": sum(hits) / len(relevant_ids),
            "mrr": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
            "ndcg": dcg / ideal, "no_result": float(not ranked)}


def percentile(values: list[float], probability: float) -> float:
    """Linear interpolation, also defined for a single measured case."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
