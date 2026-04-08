# =============================================================================
# FILE: src/mtd_engine/embed_rotator.py
# DESC: Embedding model rotation and ensemble voting for MTD-Redundancy
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# ATLAS: Mitigates AML.T0054 + AML.T0020 (combined)
#        Strategy: MTD-Redundancy — multi-model ensemble vote
# DEPS: config/mtd_config.yaml, sentence-transformers
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbedModelState:
    """State of the embedding model rotation."""

    models: list[str]
    active_model_idx: int = 0
    ensemble_mode: bool = False


class EmbedRotator:
    """Rotates embedding models and supports ensemble encoding.

    Single-model mode rotates through models to change the embedding surface.
    Ensemble mode queries all models and averages normalized embeddings,
    providing redundancy against adversarial passages optimized for one model.

    Args:
        config_path: Path to mtd_config.yaml.

    ATLAS:
        MTD-Redundancy mitigates AML.T0054 + AML.T0020 by varying the
        embedding function. Adversarial passages optimized for model A's
        embedding space are less effective when encoded by model B or C.
    """

    def __init__(self, config_path: str = "config/mtd_config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mtd_cfg = cfg["mtd"]

        self.model_names: list[str] = mtd_cfg["embed_models"]
        self._model_cache: dict[str, SentenceTransformer] = {}

        logger.info("EmbedRotator initialized: models=%s", self.model_names)

    def create_initial_state(self) -> EmbedModelState:
        """Create initial embedding model state."""
        return EmbedModelState(
            models=list(self.model_names),
            active_model_idx=0,
            ensemble_mode=False,
        )

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Lazy-load and cache a SentenceTransformer model."""
        if model_name not in self._model_cache:
            logger.info("Loading embedding model: %s", model_name)
            self._model_cache[model_name] = SentenceTransformer(model_name)
        return self._model_cache[model_name]

    def get_active_model(self, state: EmbedModelState) -> SentenceTransformer:
        """Return the currently active embedding model.

        Lazy-loads the model on first access.

        Args:
            state: Current EmbedModelState.

        Returns:
            SentenceTransformer model instance.
        """
        model_name = state.models[state.active_model_idx]
        return self._load_model(model_name)

    def get_active_model_name(self, state: EmbedModelState) -> str:
        """Return the name of the currently active model.

        Args:
            state: Current EmbedModelState.

        Returns:
            Model name string.
        """
        return state.models[state.active_model_idx]

    def rotate_model(self, state: EmbedModelState) -> EmbedModelState:
        """Rotate to the next embedding model.

        Args:
            state: Current EmbedModelState.

        Returns:
            Updated EmbedModelState with next model active.
        """
        new_idx = (state.active_model_idx + 1) % len(state.models)
        new_state = EmbedModelState(
            models=state.models,
            active_model_idx=new_idx,
            ensemble_mode=state.ensemble_mode,
        )
        # MTD-Diversity: embedding surface change
        logger.info("Embed model rotated: %s", state.models[new_idx])
        return new_state

    def encode_single(
        self,
        texts: list[str],
        state: EmbedModelState,
    ) -> np.ndarray:
        """Encode texts with the active model only.

        Args:
            texts: List of text strings to encode.
            state: Current EmbedModelState.

        Returns:
            Embedding matrix of shape (N, dim).
        """
        model = self.get_active_model(state)
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms

    def encode_ensemble(
        self,
        texts: list[str],
        state: EmbedModelState,
    ) -> np.ndarray:
        """Encode texts with ALL models and average normalized embeddings.

        If model dimensions differ, zero-pads smaller embeddings to max_dim.

        Args:
            texts: List of text strings to encode.
            state: Current EmbedModelState.

        Returns:
            Averaged embedding matrix of shape (N, max_dim).
        """
        all_embeddings: list[np.ndarray] = []
        max_dim = 0

        # Encode with each model
        for model_name in state.models:
            model = self._load_model(model_name)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            emb = emb.astype(np.float32)
            # L2 normalize per model
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            emb = emb / norms
            all_embeddings.append(emb)
            max_dim = max(max_dim, emb.shape[1])

        # Zero-pad to max_dim if dimensions differ
        padded: list[np.ndarray] = []
        for emb in all_embeddings:
            if emb.shape[1] < max_dim:
                pad_width = max_dim - emb.shape[1]
                emb = np.pad(emb, ((0, 0), (0, pad_width)), mode="constant")
            padded.append(emb)

        # MTD-Redundancy: AML.T0054 + AML.T0020 mitigation
        averaged = np.mean(padded, axis=0).astype(np.float32)

        # Re-normalize the averaged embeddings
        norms = np.linalg.norm(averaged, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return averaged / norms

    @property
    def is_loaded(self) -> bool:
        """Check if any models have been loaded."""
        return len(self._model_cache) > 0
