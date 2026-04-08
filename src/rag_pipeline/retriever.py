# =============================================================================
# FILE: src/rag_pipeline/retriever.py
# DESC: Dense / sparse / hybrid retrieval for RAG pipeline
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: config/rag_config.yaml, src/rag_pipeline/indexer.py
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.rag_pipeline.indexer import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk, score, and rank."""

    chunk: Chunk
    score: float
    rank: int


class Retriever:
    """Retrieves top-k chunks using dense, sparse, or hybrid strategy.

    Args:
        strategy: Retrieval strategy — one of 'dense', 'sparse', 'hybrid'.
        dense_index: FAISS IndexFlatIP for dense retrieval.
        sparse_index: BM25Okapi index for sparse retrieval.
        chunks: List of Chunk objects corresponding to index entries.
        encoder: SentenceTransformer model for query encoding.
        config_path: Path to rag_config.yaml.

    Raises:
        ValueError: If strategy is not one of dense/sparse/hybrid.

    ATLAS:
        Target component for AML.T0057 (RAG Database Prompting).
        Adversarial documents exploit retrieval ranking to surface poisoned content.
    """

    VALID_STRATEGIES = ("dense", "sparse", "hybrid")

    def __init__(
        self,
        strategy: str,
        dense_index: faiss.IndexFlatIP,
        sparse_index: BM25Okapi,
        chunks: list[Chunk],
        encoder: SentenceTransformer,
        config_path: str = "config/rag_config.yaml",
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {self.VALID_STRATEGIES}")

        self.strategy = strategy
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.chunks = chunks
        self.encoder = encoder

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.hybrid_alpha: float = cfg["rag"]["hybrid_alpha"]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k chunks for a given query.

        Args:
            query: User query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult ordered by score descending.
        """
        top_k = min(top_k, len(self.chunks))

        if self.strategy == "dense":
            return self._dense_retrieve(query, top_k)
        elif self.strategy == "sparse":
            return self._sparse_retrieve(query, top_k)
        else:
            return self._hybrid_retrieve(query, top_k)

    def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        # Encode and normalize query
        q_emb = self.encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        q_emb = q_emb.astype(np.float32)
        faiss.normalize_L2(q_emb)

        scores, indices = self.dense_index.search(q_emb, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue  # FAISS returns -1 for missing results
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank + 1,
            ))
        return results

    def _sparse_retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        tokenized_query = query.lower().split()
        scores = self.sparse_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        # Get all dense scores
        q_emb = self.encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        q_emb = q_emb.astype(np.float32)
        faiss.normalize_L2(q_emb)

        n = len(self.chunks)
        k_all = min(n, len(self.chunks))
        dense_scores_raw, dense_indices = self.dense_index.search(q_emb, k_all)
        dense_scores_raw = dense_scores_raw[0]
        dense_indices = dense_indices[0]

        # Build dense score array indexed by chunk position
        dense_scores = np.zeros(n, dtype=np.float32)
        for idx, score in zip(dense_indices, dense_scores_raw):
            if idx >= 0:
                dense_scores[idx] = score

        # Get sparse scores
        tokenized_query = query.lower().split()
        sparse_scores = np.array(self.sparse_index.get_scores(tokenized_query), dtype=np.float32)

        # Normalize both to [0, 1]
        dense_norm = self._normalize_scores(dense_scores.tolist())
        sparse_norm = self._normalize_scores(sparse_scores.tolist())

        # Combine: score = alpha * dense + (1-alpha) * sparse
        combined = [
            self.hybrid_alpha * d + (1 - self.hybrid_alpha) * s
            for d, s in zip(dense_norm, sparse_norm)
        ]

        top_indices = np.argsort(combined)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(combined[idx]),
                rank=rank + 1,
            ))
        return results

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1] range.

        Args:
            scores: Raw score list.

        Returns:
            Normalized score list in [0, 1].
        """
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
