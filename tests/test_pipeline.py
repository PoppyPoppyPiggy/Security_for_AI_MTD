# =============================================================================
# FILE: tests/test_pipeline.py
# DESC: Unit tests for RAGPipeline — baseline mode, batch, output fields
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/rag_pipeline/pipeline.py
# =============================================================================
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag_pipeline.pipeline import PipelineOutput, RAGPipeline, _measure_latency


class TestMeasureLatency:
    def test_returns_result_and_time(self) -> None:
        """Should return function result and positive elapsed time."""
        result, ms = _measure_latency(lambda x: x * 2, 5)
        assert result == 10
        assert ms >= 0.0


class TestPipelineBaseline:
    @patch("src.rag_pipeline.pipeline.Generator")
    def test_baseline_run(self, mock_gen_cls: MagicMock) -> None:
        """Single query in baseline mode should return non-empty answer."""
        mock_gen = MagicMock()
        mock_gen_output = MagicMock()
        mock_gen_output.answer = "A firewall filters traffic."
        mock_gen_output.prompt_used = "prompt"
        mock_gen_output.token_count = 50
        mock_gen.generate.return_value = mock_gen_output
        mock_gen_cls.return_value = mock_gen

        pipeline = RAGPipeline(mode="baseline")
        output = pipeline.run("What is a firewall?")

        assert isinstance(output, PipelineOutput)
        assert len(output.answer) > 0
        assert output.active_kb == "data/knowledge_bases/kb_clean/"
        assert output.active_retriever == "dense"

    @patch("src.rag_pipeline.pipeline.Generator")
    def test_output_fields_populated(self, mock_gen_cls: MagicMock) -> None:
        """All PipelineOutput fields should be populated."""
        mock_gen = MagicMock()
        mock_gen_output = MagicMock()
        mock_gen_output.answer = "Test answer."
        mock_gen.generate.return_value = mock_gen_output
        mock_gen_cls.return_value = mock_gen

        pipeline = RAGPipeline(mode="baseline")
        output = pipeline.run("test query")

        assert output.answer is not None
        assert output.retrieved_chunks is not None
        assert output.retrieval_latency_ms >= 0
        assert output.generation_latency_ms >= 0
        assert output.active_kb is not None
        assert output.active_retriever is not None
        assert output.active_embedder is not None

    @patch("src.rag_pipeline.pipeline.Generator")
    def test_batch_consistency(self, mock_gen_cls: MagicMock) -> None:
        """Same query run twice should return consistent answers (temp=0.0)."""
        mock_gen = MagicMock()
        mock_gen_output = MagicMock()
        mock_gen_output.answer = "Consistent answer."
        mock_gen.generate.return_value = mock_gen_output
        mock_gen_cls.return_value = mock_gen

        pipeline = RAGPipeline(mode="baseline")
        outputs = pipeline.run_batch(["What is encryption?", "What is encryption?"])

        assert len(outputs) == 2
        assert outputs[0].answer == outputs[1].answer

    @patch("src.rag_pipeline.pipeline.Generator")
    def test_custom_kb_path(self, mock_gen_cls: MagicMock) -> None:
        """Pipeline should accept custom kb_path override."""
        mock_gen_cls.return_value = MagicMock()
        mock_gen_cls.return_value.generate.return_value = MagicMock(answer="ok")

        pipeline = RAGPipeline(mode="baseline", kb_path="data/knowledge_bases/kb_clean/")
        assert pipeline.kb_path == "data/knowledge_bases/kb_clean/"
