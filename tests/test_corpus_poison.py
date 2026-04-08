# =============================================================================
# FILE: tests/test_corpus_poison.py
# DESC: Unit tests for CorpusPoisonAttacker
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/attacker/corpus_poison.py
# =============================================================================
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.attacker.corpus_poison import (
    AdversarialPassage,
    CorpusPoisonAttacker,
    InjectionReport,
)
from src.rag_pipeline.indexer import Chunk


@pytest.fixture
def attacker() -> CorpusPoisonAttacker:
    """Create attacker instance with default config."""
    return CorpusPoisonAttacker("config/attack_config.yaml")


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(text="A firewall monitors and filters network traffic based on security rules.",
              source_path="t.txt", chunk_id="c0", doc_id="d0"),
        Chunk(text="Encryption protects data by converting it to ciphertext.",
              source_path="t.txt", chunk_id="c1", doc_id="d1"),
        Chunk(text="Intrusion detection systems monitor for suspicious activity.",
              source_path="t.txt", chunk_id="c2", doc_id="d2"),
    ]


@pytest.fixture
def temp_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Create temp clean corpus and output directory."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.txt").write_text("Firewalls protect networks.")
    (corpus / "doc2.txt").write_text("VPNs encrypt traffic.")
    output = tmp_path / "output"
    return corpus, output


class TestCraftAdversarialPassage:
    def test_passage_contains_target(
        self,
        attacker: CorpusPoisonAttacker,
        sample_chunks: list[Chunk],
    ) -> None:
        """Target answer must appear in the crafted passage."""
        passage = attacker.craft_adversarial_passage(
            target_query="What is a firewall?",
            target_answer="INJECTED_ANSWER",
            existing_chunks=sample_chunks,
            attacker_level="L1",
        )
        assert isinstance(passage, AdversarialPassage)
        assert "INJECTED_ANSWER" in passage.text
        assert passage.similarity_to_query > 0.0

    def test_l2_similarity_higher(
        self,
        attacker: CorpusPoisonAttacker,
        sample_chunks: list[Chunk],
    ) -> None:
        """L2 passage should have higher similarity than L1."""
        l1 = attacker.craft_adversarial_passage(
            target_query="What is a firewall?",
            target_answer="INJECTED_ANSWER",
            existing_chunks=sample_chunks,
            attacker_level="L1",
        )
        l2 = attacker.craft_adversarial_passage(
            target_query="What is a firewall?",
            target_answer="INJECTED_ANSWER",
            existing_chunks=sample_chunks,
            attacker_level="L2",
        )
        assert l2.similarity_to_query >= l1.similarity_to_query

    def test_invalid_level_raises(
        self,
        attacker: CorpusPoisonAttacker,
        sample_chunks: list[Chunk],
    ) -> None:
        """Unsupported attacker level should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported attacker level"):
            attacker.craft_adversarial_passage("q", "a", sample_chunks, "L9")


class TestInjectAtCorpusLevel:
    def test_injection_ratio(
        self,
        attacker: CorpusPoisonAttacker,
        temp_corpus: tuple[Path, Path],
    ) -> None:
        """Injected fraction should approximate injection_ratio."""
        corpus, output = temp_corpus
        passages = [
            AdversarialPassage(text="poison text", trigger_phrase="test",
                               similarity_to_query=0.8),
        ]
        report = attacker.inject_at_corpus_level(
            str(corpus), passages, injection_ratio=0.5, output_path=str(output),
        )
        assert isinstance(report, InjectionReport)
        assert report.injected == 1
        assert report.total_docs == 3  # 2 original + 1 injected

    def test_files_created(
        self,
        attacker: CorpusPoisonAttacker,
        temp_corpus: tuple[Path, Path],
    ) -> None:
        """Output directory should contain original + injected files."""
        corpus, output = temp_corpus
        passages = [
            AdversarialPassage(text=f"poison {i}", trigger_phrase="t",
                               similarity_to_query=0.7)
            for i in range(3)
        ]
        attacker.inject_at_corpus_level(
            str(corpus), passages, injection_ratio=0.5, output_path=str(output),
        )
        all_files = list(output.iterdir())
        assert len(all_files) == 5  # 2 original + 3 injected


class TestComputeInjectionStats:
    def test_stats_format(self, attacker: CorpusPoisonAttacker) -> None:
        """Stats dict should have expected keys."""
        report = InjectionReport(total_docs=100, injected=5, injection_pct=0.05)
        stats = attacker.compute_injection_stats(report)
        assert stats["total_docs"] == 100
        assert stats["injected"] == 5
        assert stats["injection_pct"] == 0.05
        assert abs(stats["target_coverage"] - 0.05) < 1e-6
