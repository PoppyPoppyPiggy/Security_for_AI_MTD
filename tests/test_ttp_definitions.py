# =============================================================================
# FILE: tests/test_ttp_definitions.py
# DESC: Unit tests for ATLAS TTP definitions registry
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/atlas_mapper/ttp_definitions.py
# =============================================================================
from __future__ import annotations

import pytest

from src.atlas_mapper.ttp_definitions import ATLASRegistry, ATLASTechnique


class TestATLASRegistry:
    def test_all_ttps_loaded(self) -> None:
        """Registry should contain exactly 5 techniques."""
        assert len(ATLASRegistry.TECHNIQUES) == 5

    def test_get_existing(self) -> None:
        """get() should return correct technique for valid ID."""
        t = ATLASRegistry.get("AML.T0054")
        assert t is not None
        assert t.name == "False RAG Entry Injection"
        assert t.tactic == "ML Attack Staging"
        assert t.rag_relevant is True
        assert t.attack_stage == "ingestion"

    def test_get_missing(self) -> None:
        """get() should return None for unknown ID."""
        assert ATLASRegistry.get("AML.T9999") is None

    def test_rag_relevant_filter(self) -> None:
        """All returned items should have rag_relevant=True."""
        relevant = ATLASRegistry.get_rag_relevant()
        assert len(relevant) == 5
        for t in relevant:
            assert t.rag_relevant is True

    def test_stage_filter_ingestion(self) -> None:
        """Should find 2 techniques at ingestion stage."""
        ingestion = ATLASRegistry.get_by_stage("ingestion")
        assert len(ingestion) == 2
        ids = {t.id for t in ingestion}
        assert ids == {"AML.T0054", "AML.T0020"}

    def test_stage_filter_retrieval(self) -> None:
        """Should find 2 techniques at retrieval stage."""
        retrieval = ATLASRegistry.get_by_stage("retrieval")
        assert len(retrieval) == 2

    def test_stage_filter_generation(self) -> None:
        """Should find 1 technique at generation stage."""
        generation = ATLASRegistry.get_by_stage("generation")
        assert len(generation) == 1
        assert generation[0].id == "AML.T0051"

    def test_list_all(self) -> None:
        """list_all should return all 5 techniques."""
        all_ttps = ATLASRegistry.list_all()
        assert len(all_ttps) == 5
        assert all(isinstance(t, ATLASTechnique) for t in all_ttps)
