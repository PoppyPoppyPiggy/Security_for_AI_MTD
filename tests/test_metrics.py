# =============================================================================
# FILE: tests/test_metrics.py
# DESC: Unit tests for evaluation metrics — ASR, MRR, F1, latency
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/evaluation/metrics.py
# =============================================================================
from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    compute_asr,
    compute_f1,
    compute_mean_latency,
    compute_mrr,
    summarize,
)
from src.rag_pipeline.indexer import Chunk
from src.rag_pipeline.pipeline import PipelineOutput
from src.rag_pipeline.retriever import RetrievalResult


def _make_output(answer: str, doc_id: str = "d0", rank: int = 1,
                 retr_ms: float = 10.0, gen_ms: float = 50.0) -> PipelineOutput:
    """Create a minimal PipelineOutput for testing."""
    chunk = Chunk(text="test", source_path="t.txt", chunk_id="c0", doc_id=doc_id)
    return PipelineOutput(
        answer=answer,
        retrieved_chunks=[RetrievalResult(chunk=chunk, score=0.9, rank=rank)],
        retrieval_latency_ms=retr_ms,
        generation_latency_ms=gen_ms,
        active_kb="kb_clean",
        active_retriever="dense",
        active_embedder="all-MiniLM-L6-v2",
    )


class TestASR:
    def test_full_success(self) -> None:
        """All outputs contain target answer → ASR = 1.0."""
        outputs = [_make_output("INJECTED_ANSWER is here") for _ in range(5)]
        asr = compute_asr(outputs, ["q"] * 5, "INJECTED_ANSWER")
        assert asr == 1.0

    def test_no_success(self) -> None:
        """No output contains target answer → ASR = 0.0."""
        outputs = [_make_output("A firewall filters traffic.") for _ in range(5)]
        asr = compute_asr(outputs, ["q"] * 5, "INJECTED_ANSWER")
        assert asr == 0.0

    def test_partial_success(self) -> None:
        """3/5 outputs contain target → ASR = 0.6."""
        outputs = [
            _make_output("INJECTED_ANSWER"),
            _make_output("INJECTED_ANSWER"),
            _make_output("INJECTED_ANSWER"),
            _make_output("clean answer"),
            _make_output("another clean answer"),
        ]
        asr = compute_asr(outputs, ["q"] * 5, "INJECTED_ANSWER")
        assert abs(asr - 0.6) < 1e-9

    def test_case_insensitive(self) -> None:
        """ASR check should be case-insensitive."""
        outputs = [_make_output("injected_answer is present")]
        asr = compute_asr(outputs, ["q"], "INJECTED_ANSWER")
        assert asr == 1.0

    def test_empty_queries(self) -> None:
        """Empty target queries → ASR = 0.0."""
        assert compute_asr([], [], "INJECTED_ANSWER") == 0.0

    def test_asr_range(self) -> None:
        """ASR should always be in [0.0, 1.0]."""
        outputs = [_make_output("some answer") for _ in range(10)]
        asr = compute_asr(outputs, ["q"] * 10, "INJECTED_ANSWER")
        assert 0.0 <= asr <= 1.0


class TestMRR:
    def test_perfect_mrr(self) -> None:
        """First result is relevant → MRR = 1.0."""
        outputs = [_make_output("ans", doc_id="d0", rank=1)]
        mrr = compute_mrr(outputs, ["d0"], k=10)
        assert mrr == 1.0

    def test_no_relevant(self) -> None:
        """No relevant doc found → MRR = 0.0."""
        outputs = [_make_output("ans", doc_id="d0", rank=1)]
        mrr = compute_mrr(outputs, ["d999"], k=10)
        assert mrr == 0.0

    def test_empty(self) -> None:
        assert compute_mrr([], []) == 0.0


class TestF1:
    def test_exact_match(self) -> None:
        """Identical strings → F1 = 1.0."""
        assert compute_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        """No common tokens → F1 = 0.0."""
        assert compute_f1("hello", "world") == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap should give F1 between 0 and 1."""
        f1 = compute_f1("hello world foo", "hello world bar")
        assert 0.0 < f1 < 1.0

    def test_case_insensitive(self) -> None:
        """F1 should be case-insensitive."""
        assert compute_f1("Hello World", "hello world") == 1.0

    def test_empty_prediction(self) -> None:
        assert compute_f1("", "ground truth") == 0.0


class TestLatency:
    def test_mean_latency(self) -> None:
        outputs = [_make_output("a", retr_ms=10.0, gen_ms=50.0),
                   _make_output("b", retr_ms=20.0, gen_ms=60.0)]
        lat = compute_mean_latency(outputs)
        assert lat["retrieval_ms"] == 15.0
        assert lat["generation_ms"] == 55.0
        assert lat["total_ms"] == 70.0

    def test_empty(self) -> None:
        lat = compute_mean_latency([])
        assert lat["total_ms"] == 0.0


class TestSummarize:
    def test_all_keys_present(self) -> None:
        """Summarize should return all expected keys."""
        outputs = [_make_output("INJECTED_ANSWER", doc_id="d0")]
        result = summarize(outputs, ["q"], "INJECTED_ANSWER", ["d0"], ["ground truth"])
        assert "asr" in result
        assert "mrr10" in result
        assert "mean_f1" in result
        assert "latency" in result
        assert "n_queries" in result
        assert "n_target_queries" in result
