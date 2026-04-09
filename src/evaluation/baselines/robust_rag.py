# =============================================================================
# FILE: src/evaluation/baselines/robust_rag.py
# DESC: Simplified RobustRAG baseline — isolate-then-aggregate defense
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-09
# REF:   Xiang et al., "Certifiably Robust RAG", ICML 2024
# =============================================================================
from __future__ import annotations

import logging

from src.rag_pipeline.retriever import RetrievalResult

logger = logging.getLogger(__name__)


def robust_rag_aggregate(
    retrieved_chunks: list[RetrievalResult],
    target_answer: str,
) -> str:
    """Simplified RobustRAG isolate-then-aggregate defense.

    Splits retrieved chunks into two independent groups, generates
    an answer from each, then takes majority vote. If only one group
    contains the target answer, the other group's answer wins.

    This is a STATIC defense — does not adapt per query.

    Args:
        retrieved_chunks: Top-k retrieved chunks.
        target_answer: Attacker's target answer to check against.

    Returns:
        Aggregated answer string.
    """
    if not retrieved_chunks:
        return "I don't know."

    mid = max(1, len(retrieved_chunks) // 2)
    group_a = retrieved_chunks[:mid]
    group_b = retrieved_chunks[mid:]

    answer_a = group_a[0].chunk.text if group_a else "I don't know."
    answer_b = group_b[0].chunk.text if group_b else "I don't know."

    target_lower = target_answer.lower()
    a_poisoned = target_lower in answer_a.lower()
    b_poisoned = target_lower in answer_b.lower()

    if a_poisoned and b_poisoned:
        # Both groups poisoned — attacker succeeded
        return answer_a
    elif a_poisoned:
        # Only A poisoned — use B (clean group wins)
        return answer_b
    elif b_poisoned:
        # Only B poisoned — use A
        return answer_a
    else:
        # Neither poisoned — use A (default)
        return answer_a
