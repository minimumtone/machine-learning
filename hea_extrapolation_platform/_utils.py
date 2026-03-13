"""
Shared utility functions for the HEA Extrapolation Platform.
共通ユーティリティ関数

This module consolidates helper functions that were previously duplicated
across ``runner.py`` (``safe_array``) and ``workflows.py`` (``_safe_np``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_array(source: Any, dtype: str = "float64") -> np.ndarray:
    """Convert *source* to a C-contiguous numpy array.

    This is the single choke-point for every DataFrame -> numpy conversion
    in the platform.  pandas 3.0 returns F-contiguous (column-major) arrays
    from ``.values`` and ``.to_numpy()`` when the BlockManager is fragmented.
    Many C extensions (BLAS, LAPACK, scipy, sklearn) assume C-contiguous
    (row-major) layout and SIGSEGV on F-contiguous input.

    Accepts: DataFrame, Series, ndarray, or list.
    Returns: C-contiguous ndarray with requested dtype.
    """
    if isinstance(source, pd.DataFrame):
        arr = source.to_numpy(dtype=dtype, na_value=np.nan)
    elif isinstance(source, pd.Series):
        arr = source.to_numpy(dtype=dtype)
    elif isinstance(source, np.ndarray):
        arr = np.array(source, dtype=dtype)
    else:
        arr = np.asarray(source, dtype=dtype)
    return np.ascontiguousarray(arr)
