# =============================================================================
# FILE: tests/test_retriever_switcher.py
# DESC: Unit tests for RetrieverSwitcher
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/mtd_engine/retriever_switcher.py
# =============================================================================
from __future__ import annotations

import pytest

from src.mtd_engine.retriever_switcher import RetrieverSwitcher


@pytest.fixture
def switcher() -> RetrieverSwitcher:
    """Create RetrieverSwitcher with default config."""
    return RetrieverSwitcher("config/mtd_config.yaml")


class TestRetrieverSwitcher:
    def test_initial_strategy(self, switcher: RetrieverSwitcher) -> None:
        """Initial strategy should be first in sequence."""
        assert switcher.get_current_strategy() == "dense"

    def test_sequence_cycles(self, switcher: RetrieverSwitcher) -> None:
        """After cycling through all strategies, should return to first."""
        seq_len = len(switcher.sequence)
        batch_size = switcher.batch_size

        # Step through all batches to complete one full cycle
        for _ in range(seq_len * batch_size):
            switcher.step()

        assert switcher.get_current_strategy() == switcher.sequence[0]

    def test_batch_boundary(self, switcher: RetrieverSwitcher) -> None:
        """Strategy only changes at batch_size boundary."""
        first = switcher.get_current_strategy()

        # Steps within first batch should keep same strategy
        for _ in range(switcher.batch_size - 1):
            strategy = switcher.step()
            assert strategy == first

        # Next step triggers switch
        strategy = switcher.step()
        assert strategy == switcher.sequence[1]

    def test_force_switch(self, switcher: RetrieverSwitcher) -> None:
        """force_switch should immediately change strategy."""
        switcher.force_switch("sparse")
        assert switcher.get_current_strategy() == "sparse"

        switcher.force_switch("hybrid")
        assert switcher.get_current_strategy() == "hybrid"

    def test_force_switch_invalid_raises(self, switcher: RetrieverSwitcher) -> None:
        """force_switch with unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="not in sequence"):
            switcher.force_switch("nonexistent")

    def test_force_switch_resets_batch_count(self, switcher: RetrieverSwitcher) -> None:
        """force_switch should reset batch_count to 0."""
        # Ensure batch_size is large enough that steps don't auto-reset
        switcher.batch_size = 10
        switcher.step()
        switcher.step()
        assert switcher.batch_count == 2
        switcher.force_switch("sparse")
        assert switcher.batch_count == 0
