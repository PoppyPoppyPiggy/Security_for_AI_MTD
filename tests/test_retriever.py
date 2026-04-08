# =============================================================================
# FILE: tests/test_retriever.py
# DESC: Unit tests for Retriever — dense, sparse, hybrid strategies
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/rag_pipeline/retriever.py, src/rag_pipeline/indexer.py
# =============================================================================
from __future__ import annotations

import pytest

from src.rag_pipeline.indexer import DocumentIndexer
from src.rag_pipeline.retriever import Retriever, RetrievalResult


@pytest.fixture(scope="module")
def indexed_data() -> tuple:
    """Build indexes from seed corpus once for all tests."""
    indexer = DocumentIndexer(config_path="config/rag_config.yaml")
    docs = indexer.load_documents("data/knowledge_bases/kb_clean/")
    chunks = indexer.chunk_documents(docs)
    dense_index = indexer.build_dense_index(chunks)
    sparse_index = indexer.build_sparse_index(chunks)
    return dense_index, sparse_index, chunks, indexer.encoder


class TestDenseRetrieval:
    def test_returns_top_k(self, indexed_data: tuple) -> None:
        """Dense retrieval should return exactly top_k results."""
        dense_index, sparse_index, chunks, encoder = indexed_data
        retriever = Retriever("dense", dense_index, sparse_index, chunks, encoder)
        results = retriever.retrieve("What is a firewall?", top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_scores_descending(self, indexed_data: tuple) -> None:
        """Results should be ordered by score descending."""
        dense_index, sparse_index, chunks, encoder = indexed_data
        retriever = Retriever("dense", dense_index, sparse_index, chunks, encoder)
        results = retriever.retrieve("encryption algorithm", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestSparseRetrieval:
    def test_keyword_ranking(self, indexed_data: tuple) -> None:
        """Known keyword query should rank exact-match doc highly."""
        dense_index, sparse_index, chunks, encoder = indexed_data
        retriever = Retriever("sparse", dense_index, sparse_index, chunks, encoder)
        results = retriever.retrieve("firewall network security", top_k=5)
        assert len(results) == 5
        # Top result should contain "firewall"
        assert "firewall" in results[0].chunk.text.lower()


class TestHybridRetrieval:
    def test_score_range(self, indexed_data: tuple) -> None:
        """Hybrid scores should be in [0, 1] range."""
        dense_index, sparse_index, chunks, encoder = indexed_data
        retriever = Retriever("hybrid", dense_index, sparse_index, chunks, encoder)
        results = retriever.retrieve("What is encryption?", top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_different_from_dense(self, indexed_data: tuple) -> None:
        """Hybrid may produce different ranking than dense alone."""
        dense_index, sparse_index, chunks, encoder = indexed_data
        dense_ret = Retriever("dense", dense_index, sparse_index, chunks, encoder)
        hybrid_ret = Retriever("hybrid", dense_index, sparse_index, chunks, encoder)
        query = "vulnerability scanning CVE CVSS"
        dense_results = dense_ret.retrieve(query, top_k=5)
        hybrid_results = hybrid_ret.retrieve(query, top_k=5)
        # At least verify both return results (ranking may differ)
        assert len(dense_results) == 5
        assert len(hybrid_results) == 5


class TestInvalidStrategy:
    def test_raises_on_invalid(self, indexed_data: tuple) -> None:
        """Invalid strategy should raise ValueError."""
        dense_index, sparse_index, chunks, encoder = indexed_data
        with pytest.raises(ValueError, match="Invalid strategy"):
            Retriever("invalid", dense_index, sparse_index, chunks, encoder)


class TestNormalizeScores:
    def test_normalize(self) -> None:
        """Normalization should map to [0, 1]."""
        scores = [1.0, 3.0, 5.0]
        normed = Retriever._normalize_scores(scores)
        assert normed == [0.0, 0.5, 1.0]

    def test_normalize_constant(self) -> None:
        """Constant scores should normalize to all zeros."""
        scores = [2.0, 2.0, 2.0]
        normed = Retriever._normalize_scores(scores)
        assert normed == [0.0, 0.0, 0.0]
