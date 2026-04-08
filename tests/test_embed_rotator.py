# =============================================================================
# FILE: tests/test_embed_rotator.py
# DESC: Unit tests for EmbedRotator
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/mtd_engine/embed_rotator.py
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest

from src.mtd_engine.embed_rotator import EmbedModelState, EmbedRotator


@pytest.fixture
def rotator() -> EmbedRotator:
    """Create EmbedRotator with default config."""
    return EmbedRotator("config/mtd_config.yaml")


class TestEmbedRotator:
    def test_lazy_load(self, rotator: EmbedRotator) -> None:
        """Model should not be loaded at __init__, loaded on first encode."""
        assert rotator.is_loaded is False
        state = rotator.create_initial_state()
        rotator.encode_single(["test"], state)
        assert rotator.is_loaded is True

    def test_rotation_cycles(self, rotator: EmbedRotator) -> None:
        """After N rotations (N=num_models), should return to model 0."""
        state = rotator.create_initial_state()
        n_models = len(state.models)

        for _ in range(n_models):
            state = rotator.rotate_model(state)

        assert state.active_model_idx == 0

    def test_encode_single_shape(self, rotator: EmbedRotator) -> None:
        """Single encode should produce (N, dim) output."""
        state = rotator.create_initial_state()
        texts = ["Hello world", "Test sentence"]
        emb = rotator.encode_single(texts, state)
        assert emb.shape[0] == 2
        assert emb.shape[1] > 0

    def test_encode_single_normalized(self, rotator: EmbedRotator) -> None:
        """Single encode output should be L2-normalized."""
        state = rotator.create_initial_state()
        emb = rotator.encode_single(["test"], state)
        norm = np.linalg.norm(emb[0])
        assert abs(norm - 1.0) < 1e-5

    def test_ensemble_shape(self, rotator: EmbedRotator) -> None:
        """Ensemble encode should produce (N, max_dim) output."""
        state = rotator.create_initial_state()
        state = EmbedModelState(
            models=state.models,
            active_model_idx=0,
            ensemble_mode=True,
        )
        texts = ["Hello world", "Test sentence", "Another one"]
        emb = rotator.encode_ensemble(texts, state)
        assert emb.shape[0] == 3
        assert emb.shape[1] > 0

    def test_ensemble_differs_from_single(self, rotator: EmbedRotator) -> None:
        """Ensemble embedding should differ from single model embedding."""
        state = rotator.create_initial_state()
        texts = ["What is a firewall?"]
        single = rotator.encode_single(texts, state)
        ensemble = rotator.encode_ensemble(texts, state)

        # Dimensions may differ, compare up to min dim
        min_dim = min(single.shape[1], ensemble.shape[1])
        assert not np.allclose(single[:, :min_dim], ensemble[:, :min_dim], atol=1e-3)

    def test_active_model_name(self, rotator: EmbedRotator) -> None:
        """get_active_model_name should return correct model name."""
        state = rotator.create_initial_state()
        assert rotator.get_active_model_name(state) == rotator.model_names[0]
        state = rotator.rotate_model(state)
        assert rotator.get_active_model_name(state) == rotator.model_names[1]
