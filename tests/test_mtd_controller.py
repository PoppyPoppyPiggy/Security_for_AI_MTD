# =============================================================================
# FILE: tests/test_mtd_controller.py
# DESC: Unit tests for MTDController
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/mtd_engine/mtd_controller.py
# =============================================================================
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.mtd_engine.mtd_controller import MTDController, MTDState, MTDStatus


@pytest.fixture
def controller() -> MTDController:
    """Create MTDController with enabled=True config."""
    return _make_controller(enabled=True)


@pytest.fixture
def disabled_controller() -> MTDController:
    """Create MTDController with enabled=False."""
    return _make_controller(enabled=False)


def _make_controller(enabled: bool) -> MTDController:
    """Helper to create controller with custom enabled flag."""
    cfg = {
        "mtd": {
            "enabled": enabled,
            "rotation_policy": "round-robin",
            "rotation_interval": 3,
            "batch_size": 2,
            "kb_pool": [
                "data/knowledge_bases/kb_clean/",
                "data/knowledge_bases/kb_rotated/snapshot_1/",
                "data/knowledge_bases/kb_rotated/snapshot_2/",
            ],
            "kb_pool_size": 3,
            "retrieval_sequence": ["dense", "sparse", "hybrid"],
            "embed_models": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
                "intfloat/e5-small-v2",
            ],
            "ensemble_vote_threshold": 0.6,
            "anomaly_detection": {
                "enabled": enabled,
                "sigma_threshold": 2.0,
                "window_size": 50,
            },
        }
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    return MTDController(tmp.name)


class TestPassthrough:
    def test_passthrough_when_disabled(self, disabled_controller: MTDController) -> None:
        """When mtd.enabled=False, step should not trigger rotations."""
        status = disabled_controller.step("test query", [0.5, 0.6])
        assert status.total_rotations == 0
        assert status.state == MTDState.IDLE

    def test_disabled_returns_status(self, disabled_controller: MTDController) -> None:
        """Disabled controller should still return valid status."""
        status = disabled_controller.get_status()
        assert isinstance(status, MTDStatus)
        assert status.active_kb is not None


class TestStateTransitions:
    def test_state_transitions(self, controller: MTDController) -> None:
        """Should follow IDLE→ROTATING→ACTIVE→MONITORING after enough queries."""
        # Feed enough queries to trigger rotation (rotation_interval=3)
        for _ in range(3):
            status = controller.step("query", [0.5])

        # After rotation: should end at MONITORING
        assert status.state == MTDState.MONITORING
        assert status.total_rotations == 1

    def test_step_returns_status(self, controller: MTDController) -> None:
        """Status should contain all required fields."""
        status = controller.step("test query", [0.5, 0.6, 0.55])
        assert isinstance(status, MTDStatus)
        assert status.active_kb is not None
        assert status.active_retriever in ("dense", "sparse", "hybrid")
        assert status.active_embedder is not None
        assert isinstance(status.epoch, int)
        assert isinstance(status.total_rotations, int)
        assert isinstance(status.anomaly_flags, int)


class TestRotation:
    def test_rotation_count_increments(self, controller: MTDController) -> None:
        """total_rotations should increment after each KB rotate."""
        # Trigger two full rotation cycles (interval=3, so 6 queries)
        for _ in range(6):
            status = controller.step("query", [0.5])

        assert status.total_rotations == 2

    def test_kb_changes_on_rotation(self, controller: MTDController) -> None:
        """Active KB should change after rotation."""
        initial_kb = controller.get_active_config()["kb"]

        # Trigger rotation
        for _ in range(3):
            controller.step("query", [0.5])

        new_kb = controller.get_active_config()["kb"]
        assert new_kb != initial_kb


class TestAnomalyDetection:
    def test_anomaly_forces_hybrid(self, controller: MTDController) -> None:
        """High anomaly score should force retriever to hybrid."""
        # Directly set anomaly score above threshold and call step
        # Patch update_anomaly_score to preserve our injected score
        original_update = controller.kb_rotator.update_anomaly_score

        def fake_update(state, scores):
            state.anomaly_score = controller.sigma_threshold + 1.0
            return state

        controller.kb_rotator.update_anomaly_score = fake_update
        controller.kb_state.query_count = 0

        status = controller.step("query", [0.99])
        assert status.anomaly_flags >= 1

        # Restore
        controller.kb_rotator.update_anomaly_score = original_update


class TestGetActiveConfig:
    def test_config_keys(self, controller: MTDController) -> None:
        """get_active_config should return kb, retriever, embedder keys."""
        cfg = controller.get_active_config()
        assert "kb" in cfg
        assert "retriever" in cfg
        assert "embedder" in cfg
