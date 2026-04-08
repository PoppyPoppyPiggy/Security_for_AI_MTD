# =============================================================================
# FILE: tests/test_mitigation_mapper.py
# DESC: Unit tests for MitigationMapper
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/atlas_mapper/mitigation_mapper.py
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from src.atlas_mapper.mitigation_mapper import MitigationMapper, MTDMitigation
from src.atlas_mapper.ttp_definitions import ATLASRegistry


class TestMitigationMapper:
    def test_mapping_count(self) -> None:
        """Should have exactly 5 mitigations."""
        assert len(MitigationMapper.MAPPING) == 5

    def test_ttp_has_mitigation(self) -> None:
        """Every TTP in registry should have at least 1 mitigation."""
        for ttp_id in ATLASRegistry.TECHNIQUES:
            mitigations = MitigationMapper.get_mitigations_for_ttp(ttp_id)
            assert len(mitigations) >= 1, f"No mitigation for {ttp_id}"

    def test_high_effectiveness_count(self) -> None:
        """Should have exactly 2 high-effectiveness mitigations."""
        high = MitigationMapper.get_high_effectiveness()
        assert len(high) == 2

    def test_strategy_filter(self) -> None:
        """Should find mitigations by strategy name."""
        shuffling = MitigationMapper.get_mitigations_by_strategy("Shuffling")
        assert len(shuffling) == 2
        diversity = MitigationMapper.get_mitigations_by_strategy("Diversity")
        assert len(diversity) == 2
        redundancy = MitigationMapper.get_mitigations_by_strategy("Redundancy")
        assert len(redundancy) == 1

    def test_generate_table_saves_file(self, tmp_path: Path) -> None:
        """generate_table_I should create the output file."""
        save_path = str(tmp_path / "table_I.txt")
        table = MitigationMapper.generate_table_I(save_path=save_path)
        assert Path(save_path).exists()
        content = Path(save_path).read_text()
        assert "ATLAS TTP" in content
        assert "AML.T0054" in content
        assert "Shuffling" in content
        assert "SDR Triad" in content

    def test_table_contains_all_ttps(self) -> None:
        """Table should reference all 5 TTPs."""
        table = MitigationMapper.generate_table_I(save_path="/dev/null")
        for ttp_id in ATLASRegistry.TECHNIQUES:
            assert ttp_id in table
