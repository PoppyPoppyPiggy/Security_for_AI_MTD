# =============================================================================
# FILE: src/mtd_engine/kb_rotator.py
# DESC: Knowledge base rotation for MTD-Shuffling defense
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# ATLAS: Mitigates AML.T0054 (False RAG Entry Injection)
#        Strategy: MTD-Shuffling (SDR Triad)
# DEPS: config/mtd_config.yaml
# =============================================================================
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class RotationPolicy(Enum):
    """KB rotation policy."""

    ROUND_ROBIN = "round-robin"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


@dataclass
class KBPoolState:
    """State of the KB rotation pool."""

    active_kb_id: str
    pool: list[str]
    epoch: int = 0
    query_count: int = 0
    anomaly_score: float = 0.0


class KBRotator:
    """Rotates knowledge bases to disrupt persistent poisoning attacks.

    Implements three policies:
      - round-robin: Cycles through KB pool sequentially.
      - random: Randomly selects a different KB each rotation.
      - adaptive: Like random, but also triggers on anomaly detection.

    Args:
        config_path: Path to mtd_config.yaml.

    Raises:
        ValueError: If kb_pool_size < 2.

    ATLAS:
        MTD-Shuffling mitigates AML.T0054 by rotating the KB surface,
        preventing persistent poisoned entries from being consistently retrieved.
    """

    def __init__(self, config_path: str = "config/mtd_config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mtd_cfg = cfg["mtd"]

        self.policy = RotationPolicy(mtd_cfg["rotation_policy"])
        self.rotation_interval: int = mtd_cfg["rotation_interval"]
        self.kb_pool: list[str] = mtd_cfg["kb_pool"]
        self.sigma_threshold: float = mtd_cfg["anomaly_detection"]["sigma_threshold"]
        self.window_size: int = mtd_cfg["anomaly_detection"]["window_size"]

        if len(self.kb_pool) < 2:
            raise ValueError(
                f"KB pool must have at least 2 entries, got {len(self.kb_pool)}"
            )

        self._score_window: list[float] = []

        logger.info("KBRotator initialized: policy=%s, pool_size=%d, interval=%d",
                     self.policy.value, len(self.kb_pool), self.rotation_interval)

    def create_initial_state(self) -> KBPoolState:
        """Create initial KB pool state with first KB active."""
        return KBPoolState(
            active_kb_id=self.kb_pool[0],
            pool=list(self.kb_pool),
        )

    def should_rotate(self, state: KBPoolState) -> bool:
        """Check whether the KB should be rotated.

        Args:
            state: Current KBPoolState.

        Returns:
            True if rotation should occur.
        """
        interval_reached = state.query_count >= self.rotation_interval

        if self.policy == RotationPolicy.ADAPTIVE:
            anomaly_triggered = state.anomaly_score > self.sigma_threshold
            return interval_reached or anomaly_triggered

        return interval_reached

    def rotate(self, state: KBPoolState) -> KBPoolState:
        """Rotate to the next KB according to the active policy.

        Args:
            state: Current KBPoolState.

        Returns:
            Updated KBPoolState with new active KB.
        """
        old_kb = state.active_kb_id
        pool = state.pool

        if self.policy == RotationPolicy.ROUND_ROBIN:
            current_idx = pool.index(old_kb)
            next_idx = (current_idx + 1) % len(pool)
        else:
            # random and adaptive: pick a different KB
            candidates = [kb for kb in pool if kb != old_kb]
            next_idx = pool.index(random.choice(candidates))

        new_kb = pool[next_idx]
        new_state = KBPoolState(
            active_kb_id=new_kb,
            pool=pool,
            epoch=state.epoch + 1,
            query_count=0,
            anomaly_score=0.0,
        )

        # MTD-Shuffling: AML.T0054 mitigation
        logger.info("KB rotated epoch=%d: %s → %s", new_state.epoch, old_kb, new_kb)
        return new_state

    def update_anomaly_score(
        self,
        state: KBPoolState,
        retrieval_scores: list[float],
    ) -> KBPoolState:
        """Update anomaly score based on retrieval score distribution.

        Computes z-score of current batch mean vs rolling window.

        Args:
            state: Current KBPoolState.
            retrieval_scores: Scores from the latest retrieval.

        Returns:
            Updated KBPoolState with new anomaly_score.
        """
        if not retrieval_scores:
            return state

        batch_mean = float(np.mean(retrieval_scores))
        self._score_window.append(batch_mean)

        # Trim to window size
        if len(self._score_window) > self.window_size:
            self._score_window = self._score_window[-self.window_size:]

        # Need at least 3 observations for meaningful z-score
        if len(self._score_window) < 3:
            return KBPoolState(
                active_kb_id=state.active_kb_id,
                pool=state.pool,
                epoch=state.epoch,
                query_count=state.query_count,
                anomaly_score=0.0,
            )

        window_mean = float(np.mean(self._score_window))
        window_std = float(np.std(self._score_window))

        if window_std < 1e-10:
            z_score = 0.0
        else:
            z_score = abs(batch_mean - window_mean) / window_std

        # Normalize z-score to [0, 1] using sigmoid-like mapping
        anomaly_score = min(z_score / (self.sigma_threshold * 2), 1.0)

        return KBPoolState(
            active_kb_id=state.active_kb_id,
            pool=state.pool,
            epoch=state.epoch,
            query_count=state.query_count,
            anomaly_score=anomaly_score,
        )

    def get_active_kb_path(self, state: KBPoolState) -> str:
        """Return the path of the currently active KB.

        Args:
            state: Current KBPoolState.

        Returns:
            Active KB path string.
        """
        return state.active_kb_id
