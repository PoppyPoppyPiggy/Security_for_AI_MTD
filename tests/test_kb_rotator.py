# =============================================================================
# FILE: tests/test_kb_rotator.py
# DESC: Unit tests for KBRotator
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/mtd_engine/kb_rotator.py
# =============================================================================
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.mtd_engine.kb_rotator import KBPoolState, KBRotator, RotationPolicy


@pytest.fixture
def rotator() -> KBRotator:
    """Create KBRotator with default config."""
    return KBRotator("config/mtd_config.yaml")


@pytest.fixture
def small_pool_config(tmp_path: Path) -> str:
    """Create a config with only 1 KB in pool."""
    cfg = {
        "mtd": {
            "rotation_policy": "round-robin",
            "rotation_interval": 10,
            "batch_size": 3,
            "kb_pool": ["data/knowledge_bases/kb_clean/"],
            "kb_pool_size": 1,
            "retrieval_sequence": ["dense"],
            "embed_models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "ensemble_vote_threshold": 0.6,
            "anomaly_detection": {
                "enabled": False,
                "sigma_threshold": 2.0,
                "window_size": 50,
            },
        }
    }
    cfg_path = tmp_path / "mtd_config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


class TestKBRotatorInit:
    def test_pool_too_small_raises(self, small_pool_config: str) -> None:
        """ValueError if pool has only 1 KB."""
        with pytest.raises(ValueError, match="at least 2"):
            KBRotator(small_pool_config)

    def test_initial_state(self, rotator: KBRotator) -> None:
        """Initial state should have first KB active, epoch=0."""
        state = rotator.create_initial_state()
        assert state.active_kb_id == rotator.kb_pool[0]
        assert state.epoch == 0
        assert state.query_count == 0


class TestRoundRobin:
    def test_round_robin_cycles(self, rotator: KBRotator) -> None:
        """After N rotations (N=pool_size), active_kb returns to start."""
        rotator.policy = RotationPolicy.ROUND_ROBIN
        state = rotator.create_initial_state()
        start_kb = state.active_kb_id
        pool_size = len(state.pool)

        for _ in range(pool_size):
            state = rotator.rotate(state)

        assert state.active_kb_id == start_kb
        assert state.epoch == pool_size

    def test_epoch_increments(self, rotator: KBRotator) -> None:
        """Epoch should increment by 1 after each rotate()."""
        rotator.policy = RotationPolicy.ROUND_ROBIN
        state = rotator.create_initial_state()

        for i in range(5):
            state = rotator.rotate(state)
            assert state.epoch == i + 1


class TestRandomPolicy:
    def test_random_excludes_current(self, rotator: KBRotator) -> None:
        """Random policy never returns same KB twice in a row."""
        rotator.policy = RotationPolicy.RANDOM
        state = rotator.create_initial_state()

        for _ in range(20):
            old_kb = state.active_kb_id
            state = rotator.rotate(state)
            assert state.active_kb_id != old_kb


class TestAdaptivePolicy:
    def test_adaptive_triggers_on_anomaly(self, rotator: KBRotator) -> None:
        """anomaly_score > threshold should trigger rotation."""
        rotator.policy = RotationPolicy.ADAPTIVE
        state = KBPoolState(
            active_kb_id=rotator.kb_pool[0],
            pool=list(rotator.kb_pool),
            epoch=0,
            query_count=0,  # below rotation_interval
            anomaly_score=rotator.sigma_threshold + 0.1,
        )
        assert rotator.should_rotate(state) is True

    def test_no_trigger_below_threshold(self, rotator: KBRotator) -> None:
        """No rotation when both query_count and anomaly_score are below thresholds."""
        rotator.policy = RotationPolicy.ADAPTIVE
        state = KBPoolState(
            active_kb_id=rotator.kb_pool[0],
            pool=list(rotator.kb_pool),
            epoch=0,
            query_count=0,
            anomaly_score=0.1,
        )
        assert rotator.should_rotate(state) is False


class TestShouldRotate:
    def test_rotation_interval(self, rotator: KBRotator) -> None:
        """should_rotate returns True when query_count >= rotation_interval."""
        state = KBPoolState(
            active_kb_id=rotator.kb_pool[0],
            pool=list(rotator.kb_pool),
            epoch=0,
            query_count=rotator.rotation_interval,
        )
        assert rotator.should_rotate(state) is True

    def test_below_interval(self, rotator: KBRotator) -> None:
        """should_rotate returns False when below interval."""
        state = KBPoolState(
            active_kb_id=rotator.kb_pool[0],
            pool=list(rotator.kb_pool),
            epoch=0,
            query_count=rotator.rotation_interval - 1,
        )
        assert rotator.should_rotate(state) is False


class TestAnomalyScore:
    def test_update_anomaly_score(self, rotator: KBRotator) -> None:
        """Anomaly score should be computed from retrieval scores."""
        state = rotator.create_initial_state()
        # Feed normal scores to build window
        for _ in range(5):
            state = rotator.update_anomaly_score(state, [0.5, 0.6, 0.55])
        # Feed anomalous scores
        state = rotator.update_anomaly_score(state, [0.99, 0.98, 0.97])
        assert state.anomaly_score > 0.0
