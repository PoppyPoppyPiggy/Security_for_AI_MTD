# =============================================================================
# FILE: tests/test_benchmark_adapters.py
# DESC: Unit tests for benchmark adapters and extended evaluation
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/evaluation/benchmark_adapter.py,
#       src/evaluation/extended_evaluator.py,
#       src/evaluation/comparative_table.py
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.benchmark_adapter import (
    BenchmarkQuery,
    FaithfulnessResult,
    RAGCheckerAdapter,
    RAGSecBenchAdapter,
    SafeRAGAdapter,
    RAGSECBENCH_ATTACK_CATEGORIES,
    SAFERAG_DIMENSIONS,
    ATLAS_TO_SAFERAG,
)
from src.evaluation.extended_evaluator import compute_narr, compute_defense_coverage
from src.evaluation.comparative_table import (
    generate_table_II,
    BASELINE_FRAMEWORKS,
)
from src.rag_pipeline.pipeline import PipelineOutput
from src.rag_pipeline.retriever import RetrievalResult
from src.rag_pipeline.indexer import Chunk


# ---------------------------------------------------------------------------
# RAGSecBench Adapter Tests
# ---------------------------------------------------------------------------

class TestRAGSecBenchAdapter:
    def test_synthetic_generates_all_categories(self) -> None:
        """Synthetic fallback should cover all 13 attack categories."""
        adapter = RAGSecBenchAdapter()
        queries = adapter.load_queries()
        categories = {q.attack_category for q in queries}
        assert categories == set(RAGSECBENCH_ATTACK_CATEGORIES)

    def test_synthetic_query_count(self) -> None:
        """Should generate 3 queries per category (13 × 3 = 39)."""
        adapter = RAGSecBenchAdapter()
        queries = adapter.load_queries()
        assert len(queries) == 39

    def test_query_format(self) -> None:
        """Each query should be a valid BenchmarkQuery."""
        adapter = RAGSecBenchAdapter()
        queries = adapter.load_queries()
        for q in queries:
            assert isinstance(q, BenchmarkQuery)
            assert len(q.query) > 0
            assert len(q.ground_truth) > 0
            assert q.source_benchmark.startswith("ragsecbench")

    def test_get_attack_categories(self) -> None:
        """Should return the full list of 13 categories."""
        adapter = RAGSecBenchAdapter()
        cats = adapter.get_attack_categories()
        assert len(cats) == 13

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Should load queries from a JSON file."""
        import json
        data = [
            {"query": "test q", "target_answer": "TA", "ground_truth": "GT",
             "attack_category": "naive_poisoning"},
        ]
        path = tmp_path / "ragsecbench.json"
        path.write_text(json.dumps(data))
        adapter = RAGSecBenchAdapter(data_path=str(path))
        queries = adapter.load_queries()
        assert len(queries) == 1
        assert queries[0].query == "test q"
        assert queries[0].source_benchmark == "ragsecbench"


# ---------------------------------------------------------------------------
# SafeRAG Adapter Tests
# ---------------------------------------------------------------------------

class TestSafeRAGAdapter:
    def test_synthetic_dimensions(self) -> None:
        """Synthetic fallback should cover all 4 SafeRAG dimensions."""
        adapter = SafeRAGAdapter()
        queries = adapter.load_queries()
        dims = {q.attack_category for q in queries}
        assert dims == set(SAFERAG_DIMENSIONS)

    def test_synthetic_query_count(self) -> None:
        """Should generate 5 queries per dimension (4 × 5 = 20)."""
        adapter = SafeRAGAdapter()
        queries = adapter.load_queries()
        assert len(queries) == 20

    def test_atlas_coverage_mapping(self) -> None:
        """ATLAS TTP to SafeRAG dimension mapping should exist for all TTPs."""
        adapter = SafeRAGAdapter()
        coverage = adapter.get_atlas_coverage()
        assert "AML.T0054" in coverage
        assert "AML.T0051" in coverage
        assert len(coverage) == 5

    def test_get_dimensions(self) -> None:
        """Should return all 4 dimensions."""
        adapter = SafeRAGAdapter()
        assert len(adapter.get_dimensions()) == 4


# ---------------------------------------------------------------------------
# RAGChecker Adapter Tests
# ---------------------------------------------------------------------------

class TestRAGCheckerAdapter:
    def test_supported_claim(self) -> None:
        """Claim matching context should be classified as supported."""
        checker = RAGCheckerAdapter()
        result = checker.evaluate(
            answer="A firewall monitors and filters network traffic.",
            context_texts=["A firewall is a device that monitors network traffic and filters packets."],
        )
        assert result.faithfulness_score > 0.0
        assert result.supported_claims > 0

    def test_hallucinated_claim(self) -> None:
        """Claim not in context should be classified as hallucinated."""
        checker = RAGCheckerAdapter()
        result = checker.evaluate(
            answer="Quantum computing uses qubits for parallel processing.",
            context_texts=["A firewall monitors network traffic."],
        )
        assert result.hallucination_rate > 0.0
        assert result.hallucinated_claims > 0

    def test_empty_answer(self) -> None:
        """Empty answer should return perfect faithfulness."""
        checker = RAGCheckerAdapter()
        result = checker.evaluate(answer="", context_texts=["some context"])
        assert result.faithfulness_score == 1.0
        assert result.total_claims == 0

    def test_batch_evaluation(self) -> None:
        """Batch evaluation should aggregate claim counts."""
        checker = RAGCheckerAdapter()
        result = checker.evaluate_batch(
            answers=[
                "A firewall filters network traffic.",
                "Quantum computers use qubits.",
            ],
            contexts=[
                ["Firewalls filter and monitor network traffic."],
                ["Encryption protects data in transit."],
            ],
        )
        assert isinstance(result, FaithfulnessResult)
        assert result.total_claims >= 2

    def test_faithfulness_result_fields(self) -> None:
        """FaithfulnessResult should have all expected fields."""
        checker = RAGCheckerAdapter()
        result = checker.evaluate(
            answer="Firewalls protect networks from unauthorized access.",
            context_texts=["Firewalls establish a barrier to protect networks."],
        )
        assert hasattr(result, "total_claims")
        assert hasattr(result, "supported_claims")
        assert hasattr(result, "contradicted_claims")
        assert hasattr(result, "hallucinated_claims")
        assert hasattr(result, "faithfulness_score")
        assert hasattr(result, "hallucination_rate")
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert 0.0 <= result.hallucination_rate <= 1.0


# ---------------------------------------------------------------------------
# Extended Evaluator Tests
# ---------------------------------------------------------------------------

class TestNARR:
    def test_narr_basic(self) -> None:
        """NARR = (1.0 - 0.4) / 1.0 = 0.6."""
        assert compute_narr(1.0, 0.4) == 0.6

    def test_narr_perfect_defense(self) -> None:
        """NARR should be 1.0 when MTD reduces ASR to 0."""
        assert compute_narr(0.9, 0.0) == 1.0

    def test_narr_no_improvement(self) -> None:
        """NARR should be 0.0 when MTD doesn't help."""
        assert compute_narr(0.5, 0.5) == 0.0

    def test_narr_zero_baseline(self) -> None:
        """NARR should be 0.0 when baseline ASR is 0 (no attack)."""
        assert compute_narr(0.0, 0.0) == 0.0

    def test_narr_negative_clamp(self) -> None:
        """NARR should clamp to 0.0 if MTD makes things worse."""
        assert compute_narr(0.3, 0.5) == 0.0


class TestDefenseCoverage:
    def _make_output(self, answer: str) -> PipelineOutput:
        chunk = Chunk(text="ctx", source_path="t", chunk_id="c0", doc_id="d0")
        return PipelineOutput(
            answer=answer,
            retrieved_chunks=[RetrievalResult(chunk=chunk, score=0.9, rank=1)],
            retrieval_latency_ms=1.0,
            generation_latency_ms=1.0,
            active_kb="kb",
            active_retriever="dense",
            active_embedder="model",
        )

    def test_full_coverage(self) -> None:
        """All categories defended (ASR < 0.5) → coverage = 1.0."""
        queries = [
            BenchmarkQuery(query="q1", target_answer="TA", ground_truth="",
                           attack_category="cat_a", source_benchmark="test"),
            BenchmarkQuery(query="q2", target_answer="TA", ground_truth="",
                           attack_category="cat_b", source_benchmark="test"),
        ]
        outputs = [
            self._make_output("clean answer"),
            self._make_output("another clean answer"),
        ]
        result = compute_defense_coverage(queries, outputs, "TA")
        assert result["defense_coverage_rate"] == 1.0
        assert result["covered_categories"] == 2

    def test_partial_coverage(self) -> None:
        """One category fully attacked → coverage < 1.0."""
        queries = [
            BenchmarkQuery(query="q1", target_answer="TA", ground_truth="",
                           attack_category="cat_a", source_benchmark="test"),
            BenchmarkQuery(query="q2", target_answer="TA", ground_truth="",
                           attack_category="cat_b", source_benchmark="test"),
        ]
        outputs = [
            self._make_output("TA is the answer"),  # hit
            self._make_output("clean answer"),       # miss
        ]
        result = compute_defense_coverage(queries, outputs, "TA")
        assert result["covered_categories"] == 1
        assert result["total_categories"] == 2


# ---------------------------------------------------------------------------
# Comparative Table Tests
# ---------------------------------------------------------------------------

class TestComparativeTable:
    def test_baseline_frameworks_count(self) -> None:
        """Should have 6 baseline frameworks."""
        assert len(BASELINE_FRAMEWORKS) == 6

    def test_generate_table_creates_files(self, tmp_path: Path) -> None:
        """generate_table_II should create ASCII and LaTeX files."""
        ascii_path = str(tmp_path / "table_II.txt")
        latex_path = str(tmp_path / "table_II.tex")
        table = generate_table_II(
            our_results=None,
            save_ascii=ascii_path,
            save_latex=latex_path,
        )
        assert Path(ascii_path).exists()
        assert Path(latex_path).exists()
        assert "ATLAS-MTD-RAG" in table
        assert "RobustRAG" in table

    def test_table_with_results(self, tmp_path: Path) -> None:
        """Table should incorporate experiment results when provided."""
        results = {
            "mtd": {
                "metrics": {"asr": 0.20},
                "faithfulness": {"score": 0.85},
            },
            "narr": 0.80,
        }
        table = generate_table_II(
            our_results=results,
            save_ascii=str(tmp_path / "t.txt"),
            save_latex=str(tmp_path / "t.tex"),
        )
        assert "0.20" in table
        assert "0.85" in table

    def test_latex_format(self, tmp_path: Path) -> None:
        """LaTeX output should have proper tabular environment."""
        generate_table_II(
            save_ascii=str(tmp_path / "t.txt"),
            save_latex=str(tmp_path / "t.tex"),
        )
        latex = Path(tmp_path / "t.tex").read_text()
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\checkmark" in latex
