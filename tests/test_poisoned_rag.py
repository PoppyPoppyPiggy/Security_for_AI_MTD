# =============================================================================
# FILE: tests/test_poisoned_rag.py
# DESC: Unit tests for PoisonedRAGAttacker
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/attacker/poisoned_rag.py
# =============================================================================
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.attacker.poisoned_rag import PoisonDoc, PoisonedRAGAttacker


@pytest.fixture
def attacker() -> PoisonedRAGAttacker:
    """Create attacker instance with default config."""
    return PoisonedRAGAttacker("config/attack_config.yaml")


@pytest.fixture
def temp_kb(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary clean KB and output directory."""
    kb_clean = tmp_path / "kb_clean"
    kb_clean.mkdir()
    (kb_clean / "doc1.txt").write_text("Firewalls filter network traffic.")
    (kb_clean / "doc2.txt").write_text("Encryption protects data in transit.")

    kb_poisoned = tmp_path / "kb_poisoned"
    return kb_clean, kb_poisoned


class TestCraftPoisonDocs:
    def test_naive_doc_contains_target(self, attacker: PoisonedRAGAttacker) -> None:
        """Naive poison docs must contain the target answer."""
        docs = attacker.craft_poison_docs(
            target_query="What is a firewall?",
            target_answer="INJECTED_ANSWER",
            num_docs=3,
            strategy="naive",
        )
        assert len(docs) == 3
        for doc in docs:
            assert "INJECTED_ANSWER" in doc.text
            assert doc.is_poisoned is True
            assert doc.target_answer == "INJECTED_ANSWER"

    def test_naive_returns_correct_count(self, attacker: PoisonedRAGAttacker) -> None:
        """Should return exactly num_docs documents."""
        docs = attacker.craft_poison_docs(
            target_query="test query",
            target_answer="test answer",
            num_docs=7,
            strategy="naive",
        )
        assert len(docs) == 7

    def test_optimized_doc_contains_target(self, attacker: PoisonedRAGAttacker) -> None:
        """Optimized poison docs must contain the target answer."""
        docs = attacker.craft_poison_docs(
            target_query="What is a firewall?",
            target_answer="INJECTED_ANSWER",
            num_docs=2,
            strategy="optimized",
        )
        assert len(docs) == 2
        for doc in docs:
            assert "INJECTED_ANSWER" in doc.text
            assert doc.is_poisoned is True

    def test_invalid_strategy_raises(self, attacker: PoisonedRAGAttacker) -> None:
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            attacker.craft_poison_docs("q", "a", 1, strategy="unknown")


class TestInjectIntoKB:
    def test_inject_creates_files(
        self,
        attacker: PoisonedRAGAttacker,
        temp_kb: tuple[Path, Path],
    ) -> None:
        """Output dir should have original + poison doc files after injection."""
        kb_clean, kb_poisoned = temp_kb
        poison_docs = attacker.craft_poison_docs(
            "What is a firewall?", "INJECTED_ANSWER", num_docs=3, strategy="naive",
        )
        attacker.inject_into_kb(str(kb_clean), poison_docs, str(kb_poisoned))

        # 2 original .txt + 3 poison .txt + 3 poison .json = 8 files
        all_files = list(kb_poisoned.iterdir())
        assert len(all_files) == 8

        txt_files = [f for f in all_files if f.suffix == ".txt"]
        assert len(txt_files) == 5  # 2 original + 3 poison

    def test_poison_doc_metadata(
        self,
        attacker: PoisonedRAGAttacker,
        temp_kb: tuple[Path, Path],
    ) -> None:
        """Sidecar .json files must have is_poisoned=True."""
        kb_clean, kb_poisoned = temp_kb
        poison_docs = attacker.craft_poison_docs(
            "test", "TARGET", num_docs=2, strategy="naive",
        )
        attacker.inject_into_kb(str(kb_clean), poison_docs, str(kb_poisoned))

        json_files = sorted(kb_poisoned.glob("poison_*.json"))
        assert len(json_files) == 2
        for jf in json_files:
            meta = json.loads(jf.read_text())
            assert meta["is_poisoned"] is True
            assert meta["target_answer"] == "TARGET"


class TestEvaluateInjection:
    def test_asr_computation(self, attacker: PoisonedRAGAttacker) -> None:
        """ASR should correctly compute success fraction."""
        from unittest.mock import MagicMock

        outputs = []
        for answer in ["INJECTED_ANSWER is correct", "Real answer", "INJECTED_ANSWER"]:
            mock_output = MagicMock()
            mock_output.answer = answer
            outputs.append(mock_output)

        asr = attacker.evaluate_injection(outputs, "INJECTED_ANSWER")
        assert abs(asr - 2.0 / 3.0) < 1e-6

    def test_asr_empty(self, attacker: PoisonedRAGAttacker) -> None:
        """ASR of empty outputs should be 0."""
        assert attacker.evaluate_injection([], "anything") == 0.0
