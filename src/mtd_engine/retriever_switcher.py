# =============================================================================
# FILE: src/mtd_engine/retriever_switcher.py
# DESC: Retrieval strategy rotation for MTD-Diversity defense
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# ATLAS: Mitigates AML.T0020 (Poison Training Data)
#        Strategy: MTD-Diversity — retrieval strategy rotation
# REF:   Semantic Chameleon (Thornton 2026): hybrid retrieval → ASR 0%
# DEPS: config/mtd_config.yaml
# =============================================================================
from __future__ import annotations

import logging

from src.utils import safe_load_config

logger = logging.getLogger(__name__)


class RetrieverSwitcher:
    """Cycles retrieval strategies to disrupt retriever-aware attacks.

    By rotating between dense, sparse, and hybrid retrieval, poisoned
    documents optimized for one retrieval method are less likely to be
    consistently ranked highly.

    Args:
        config_path: Path to mtd_config.yaml.

    ATLAS:
        MTD-Diversity mitigates AML.T0020 by changing the retrieval surface.
        Adversarial passages optimized for dense retrieval may fail under
        sparse or hybrid strategies.
    """

    def __init__(self, config_path: str = "config/mtd_config.yaml") -> None:
        cfg = safe_load_config(config_path)
        mtd_cfg = cfg["mtd"]

        self.sequence: list[str] = mtd_cfg["retrieval_sequence"]
        self.batch_size: int = mtd_cfg["batch_size"]
        self.current_idx: int = 0
        self.batch_count: int = 0

        logger.info("RetrieverSwitcher initialized: sequence=%s, batch_size=%d",
                     self.sequence, self.batch_size)

    def get_current_strategy(self) -> str:
        """Return the currently active retrieval strategy.

        Returns:
            Strategy string: 'dense', 'sparse', or 'hybrid'.
        """
        return self.sequence[self.current_idx % len(self.sequence)]

    def step(self, query_count: int = 1) -> str:
        """Advance the switcher by one step, rotating at batch boundaries.

        Args:
            query_count: Number of queries processed (used for logging).

        Returns:
            The active retrieval strategy after this step.
        """
        self.batch_count += 1

        if self.batch_count >= self.batch_size:
            self.current_idx = (self.current_idx + 1) % len(self.sequence)
            self.batch_count = 0
            # MTD-Diversity: AML.T0020 mitigation
            logger.info("Retriever switched: %s", self.get_current_strategy())

        return self.get_current_strategy()

    def force_switch(self, target_strategy: str) -> None:
        """Force immediate switch to a specific strategy.

        Used by MTDController on anomaly detection to jump to hybrid.

        Args:
            target_strategy: Target strategy to switch to.

        Raises:
            ValueError: If target_strategy is not in the sequence.
        """
        if target_strategy not in self.sequence:
            raise ValueError(
                f"Strategy '{target_strategy}' not in sequence {self.sequence}"
            )
        self.current_idx = self.sequence.index(target_strategy)
        self.batch_count = 0
        logger.info("Retriever force-switched to: %s", target_strategy)
