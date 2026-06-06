"""Shared utilities: logging setup, path helpers, type guards."""

import logging
import sys
from pathlib import Path
from typing import List


def setup_logging(verbosity: int = 1) -> None:
    """Configure the ``t2vasp`` logger hierarchy.

    Parameters
    ----------
    verbosity : int
        0 = WARNING, 1 = INFO, 2+ = DEBUG.
    """
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger("t2vasp")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def find_calc_dirs(base: str | Path, markers: List[str] | None = None) -> List[Path]:
    """Return subdirectories that look like VASP calculation folders.

    A directory is included if it contains at least one of *markers*
    (default: ``OUTCAR``, ``vasprun.xml``).
    """
    markers = markers or ["OUTCAR", "vasprun.xml"]
    base = Path(base)
    dirs: List[Path] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and any((child / m).is_file() for m in markers):
            dirs.append(child)
    return dirs


def safe_float(val: str, default: float = float("nan")) -> float:
    """Convert *val* to float; return *default* on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
