# =============================================================================
# FILE: src/utils.py
# DESC: Shared utilities — config loading, path helpers
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def safe_load_config(path: str) -> dict:
    """Load a YAML config file with existence check.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or not valid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    return cfg
