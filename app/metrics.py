from __future__ import annotations
import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of all known relevant chunk IDs retrieved in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant_ids or k < 1:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, item in enumerate(retrieved_ids[:k], start=1)
        if item in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
