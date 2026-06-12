"""Configuration loading utilities."""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "defaults.yaml"


def load_config(user_path: str | None = None) -> Dict[str, Any]:
    """Load configuration from defaults.yaml, optionally overridden by *user_path*.

    Parameters
    ----------
    user_path : str or None
        Path to a user-supplied YAML that overrides default values.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    with open(_DEFAULT_CONFIG_PATH, "r") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    if user_path and os.path.isfile(user_path):
        with open(user_path, "r") as fh:
            user_cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user_cfg)
        logger.info("User config loaded from %s", user_path)

    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
