# =============================================================================
# FILE: src/mtd_engine/mtd_controller.py
# DESC: Unified MTD orchestrator for all RAG-layer defense mechanisms
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# ATLAS: Unified mitigation orchestrator for AML.T0054, AML.T0020, AML.T0051
# DEPS: config/mtd_config.yaml,
#       src/mtd_engine/kb_rotator.py,
#       src/mtd_engine/retriever_switcher.py,
#       src/mtd_engine/embed_rotator.py
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import yaml

from src.mtd_engine.embed_rotator import EmbedRotator
from src.mtd_engine.kb_rotator import KBRotator
from src.mtd_engine.retriever_switcher import RetrieverSwitcher

logger = logging.getLogger(__name__)


class MTDState(Enum):
    """MTD engine state machine states."""

    IDLE = "idle"
    ROTATING = "rotating"
    ACTIVE = "active"
    MONITORING = "monitoring"


@dataclass
class MTDStatus:
    """Snapshot of current MTD engine status."""

    state: MTDState
    active_kb: str
    active_retriever: str
    active_embedder: str
    epoch: int
    total_rotations: int
    anomaly_flags: int


class MTDController:
    """Orchestrates KB rotation, retriever switching, and embedding rotation.

    Coordinates all MTD defense mechanisms and manages the state machine:
    IDLE → ROTATING → ACTIVE → MONITORING → (loop)

    When mtd.enabled=False, operates in passthrough mode with no rotations.

    Args:
        config_path: Path to mtd_config.yaml.

    ATLAS:
        Unified mitigation for:
        - AML.T0054 (KB rotation — Shuffling)
        - AML.T0020 (Retriever switching — Diversity)
        - AML.T0051 (Embedding rotation — Redundancy)
    """

    def __init__(self, config_path: str = "config/mtd_config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mtd_cfg = cfg["mtd"]

        self.enabled: bool = mtd_cfg["enabled"]
        self.rotation_interval: int = mtd_cfg["rotation_interval"]
        self.anomaly_enabled: bool = mtd_cfg["anomaly_detection"]["enabled"]
        self.sigma_threshold: float = mtd_cfg["anomaly_detection"]["sigma_threshold"]

        if not self.enabled:
            logger.warning("MTD is disabled — operating in passthrough mode")

        # Init sub-controllers
        self.kb_rotator = KBRotator(config_path)
        self.retriever_switcher = RetrieverSwitcher(config_path)
        self.embed_rotator = EmbedRotator(config_path)

        # Init states
        self.kb_state = self.kb_rotator.create_initial_state()
        self.embed_state = self.embed_rotator.create_initial_state()

        # Status tracking
        self._state = MTDState.IDLE
        self._total_rotations: int = 0
        self._anomaly_flags: int = 0
        self._query_counter: int = 0

        logger.info("MTDController initialized: enabled=%s", self.enabled)

    def step(
        self,
        query: str,
        retrieval_scores: list[float],
    ) -> MTDStatus:
        """Main per-query hook called by RAGPipeline.

        Executes the MTD decision logic:
        1. Update anomaly score
        2. Check KB rotation trigger
        3. Step retriever switcher
        4. Force hybrid on anomaly
        5. Rotate embedding model at interval

        Args:
            query: The current query string.
            retrieval_scores: Scores from the latest retrieval.

        Returns:
            Updated MTDStatus snapshot.
        """
        if not self.enabled:
            return self.get_status()

        self._query_counter += 1
        self.kb_state.query_count += 1

        # 1. Update anomaly score
        self.kb_state = self.kb_rotator.update_anomaly_score(
            self.kb_state, retrieval_scores,
        )

        # 2. Check KB rotation
        if self.kb_rotator.should_rotate(self.kb_state):
            self._transition(MTDState.ROTATING)
            self.kb_state = self.kb_rotator.rotate(self.kb_state)
            self._total_rotations += 1
            self._transition(MTDState.ACTIVE)

        # 3. Step retriever switcher
        self.retriever_switcher.step(self._query_counter)

        # 4. Force hybrid on anomaly detection
        if (self.anomaly_enabled
                and self.kb_state.anomaly_score > self.sigma_threshold):
            self.retriever_switcher.force_switch("hybrid")
            self._anomaly_flags += 1
            logger.warning("Anomaly detected: score=%.4f, forcing hybrid retrieval",
                           self.kb_state.anomaly_score)

        # 5. Rotate embedding model at rotation interval
        if self._query_counter % self.rotation_interval == 0:
            self.embed_state = self.embed_rotator.rotate_model(self.embed_state)

        self._transition(MTDState.MONITORING)
        return self.get_status()

    def _transition(self, new_state: MTDState) -> None:
        """Transition the state machine to a new state.

        Args:
            new_state: Target MTDState.
        """
        logger.debug("MTD state: %s → %s", self._state.value, new_state.value)
        self._state = new_state

    def get_status(self) -> MTDStatus:
        """Return current MTD status snapshot.

        Returns:
            MTDStatus with all current state information.
        """
        return MTDStatus(
            state=self._state,
            active_kb=self.kb_rotator.get_active_kb_path(self.kb_state),
            active_retriever=self.retriever_switcher.get_current_strategy(),
            active_embedder=self.embed_rotator.get_active_model_name(self.embed_state),
            epoch=self.kb_state.epoch,
            total_rotations=self._total_rotations,
            anomaly_flags=self._anomaly_flags,
        )

    def get_active_config(self) -> dict[str, str]:
        """Return the active MTD configuration for pipeline use.

        Returns:
            Dict with keys: kb, retriever, embedder.
        """
        return {
            "kb": self.kb_rotator.get_active_kb_path(self.kb_state),
            "retriever": self.retriever_switcher.get_current_strategy(),
            "embedder": self.embed_rotator.get_active_model_name(self.embed_state),
        }
