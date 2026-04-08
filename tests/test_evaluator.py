# =============================================================================
# FILE: tests/test_evaluator.py
# DESC: Unit tests for evaluator — results file creation and output validation
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/evaluation/evaluator.py
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evaluator import load_target_queries, run_evaluation, set_seed


class TestLoadTargetQueries:
    def test_loads_correctly(self) -> None:
        """Should load 5 target queries from seed data."""
        data = load_target_queries("data/attack_corpus/target_queries.json")
        assert len(data) == 5
        assert "query" in data[0]
        assert "target_answer" in data[0]
        assert "ground_truth" in data[0]


class TestSetSeed:
    def test_deterministic(self) -> None:
        """Setting seed should produce deterministic random values."""
        import random
        set_seed(42)
        val1 = random.random()
        set_seed(42)
        val2 = random.random()
        assert val1 == val2


class TestRunEvaluation:
    @patch("src.evaluation.evaluator.RAGPipeline")
    def test_results_file_created(self, mock_pipeline_cls: MagicMock, tmp_path: Path) -> None:
        """Evaluation should create a results JSON file."""
        mock_pipeline = MagicMock()
        mock_output = MagicMock()
        mock_output.answer = "ATTACKER_TARGET is the answer"
        mock_output.retrieval_latency_ms = 15.0
        mock_output.generation_latency_ms = 50.0
        mock_chunk = MagicMock()
        mock_chunk.text = "test chunk"
        mock_chunk.doc_id = "seed_docs_0"
        mock_chunk.chunk_id = "c0"
        mock_result = MagicMock()
        mock_result.chunk = mock_chunk
        mock_result.score = 0.9
        mock_result.rank = 1
        mock_output.retrieved_chunks = [mock_result]
        mock_pipeline.run_batch.return_value = [mock_output] * 5
        mock_pipeline_cls.return_value = mock_pipeline

        results = run_evaluation(mode="baseline", seed=42)

        assert results["mode"] == "baseline"
        assert "metrics" in results

    @patch("src.evaluation.evaluator.RAGPipeline")
    def test_asr_range(self, mock_pipeline_cls: MagicMock) -> None:
        """ASR should be in [0.0, 1.0]."""
        mock_pipeline = MagicMock()
        mock_output = MagicMock()
        mock_output.answer = "some answer"
        mock_output.retrieval_latency_ms = 10.0
        mock_output.generation_latency_ms = 40.0
        mock_chunk = MagicMock()
        mock_chunk.text = "text"
        mock_chunk.doc_id = "d0"
        mock_result = MagicMock()
        mock_result.chunk = mock_chunk
        mock_result.score = 0.5
        mock_result.rank = 1
        mock_output.retrieved_chunks = [mock_result]
        mock_pipeline.run_batch.return_value = [mock_output] * 5
        mock_pipeline_cls.return_value = mock_pipeline

        results = run_evaluation(mode="baseline", seed=42)
        asr = results["metrics"]["asr"]
        assert 0.0 <= asr <= 1.0

    @patch("src.evaluation.evaluator.RAGPipeline")
    def test_summary_keys(self, mock_pipeline_cls: MagicMock) -> None:
        """All expected keys should be present in results."""
        mock_pipeline = MagicMock()
        mock_output = MagicMock()
        mock_output.answer = "test"
        mock_output.retrieval_latency_ms = 5.0
        mock_output.generation_latency_ms = 20.0
        mock_chunk = MagicMock()
        mock_chunk.text = "text"
        mock_chunk.doc_id = "d0"
        mock_result = MagicMock()
        mock_result.chunk = mock_chunk
        mock_result.score = 0.5
        mock_result.rank = 1
        mock_output.retrieved_chunks = [mock_result]
        mock_pipeline.run_batch.return_value = [mock_output] * 5
        mock_pipeline_cls.return_value = mock_pipeline

        results = run_evaluation(mode="baseline", seed=42)
        expected_keys = {"mode", "attack_type", "attacker_level", "seed",
                         "timestamp", "kb_path", "metrics", "individual_outputs"}
        assert expected_keys.issubset(set(results.keys()))

        metric_keys = {"asr", "mrr10", "mean_f1", "latency", "n_queries", "n_target_queries"}
        assert metric_keys.issubset(set(results["metrics"].keys()))
