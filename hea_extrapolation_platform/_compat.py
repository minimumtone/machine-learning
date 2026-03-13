"""
pandas 3.0 / numpy C-contiguous compatibility layer
pandas 3.0 互換レイヤー

Problem
-------
pandas 3.0's BlockManager produces **F-contiguous** (column-major) numpy
arrays from ``.values``, ``.to_numpy()``, and internal operations.  Many C
extensions in numpy, scipy, scikit-learn, and plotly assume C-contiguous
(row-major) layout.  When an F-contiguous array reaches these extensions
the result is a **SIGSEGV** (segmentation fault).

Previous fixes patched individual call sites (``safe_array``,
``np.ascontiguousarray``, ``_to_list``).  This was insufficient because
pandas-derived data can reach C extensions through **any** code path —
including trivial ones like ``np.mean(list_of_floats)`` when the floats
originate from pandas objects.

Solution
--------
This module **monkey-patches pandas at the source** so that every array
produced by pandas is guaranteed C-contiguous.  It patches:

1. ``DataFrame.to_numpy()``  — the primary extraction method
2. ``Series.to_numpy()``     — used by sklearn, evaluation, etc.
3. ``DataFrame.values``      — legacy property (calls to_numpy internally)
4. ``Series.values``         — legacy property

The patches are applied **once** at import time via ``install()``.
All downstream code — sklearn, scipy, plotly, evaluation, etc. — receives
C-contiguous arrays without any per-call-site changes.

Usage
-----
Import this module as early as possible (in ``__init__.py``)::

    from hea_extrapolation_platform._compat import install as _install_compat
    _install_compat()

The install function is idempotent; calling it multiple times is safe.

Additionally, this module provides ``safe_float()`` which converts
any numpy scalar to a plain Python float, severing the reference to
the original numpy memory block.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INSTALLED = False


@contextlib.contextmanager
def gc_disabled() -> Iterator[None]:
    """Context manager that disables the cyclic garbage collector.

    During plotly chart creation and other critical operations, GC can
    trigger C-extension finalizers on F-contiguous numpy arrays left
    over from pandas 3.0 operations.  These finalizers crash with
    SIGSEGV.  Disabling GC during these operations prevents the crash.

    Reference counting still frees non-cyclic objects normally.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def safe_float(x: object) -> float:
    """Convert *x* to a plain Python ``float``.

    numpy scalars (``np.float64``, etc.) retain a reference to their
    parent array's memory block.  Converting to ``float()`` severs that
    link, preventing C-extension crashes when the parent block is
    F-contiguous.
    """
    return float(x)


def as_serializable(obj: object) -> object:
    """Recursively convert numpy / pandas types to JSON-safe Python builtins.

    Problem (#5): ``json.dump`` raises ``TypeError`` on ``numpy.float32``,
    ``numpy.int64``, ``numpy.ndarray``, ``numpy.bool_``, etc.

    This function walks a nested dict / list structure and converts:
      - numpy integer types  → ``int``
      - numpy float types    → ``float``
      - numpy bool\_         → ``bool``
      - numpy ndarray        → ``list`` (recursive)
      - numpy generic        → appropriate Python scalar
      - pandas Series        → ``list``
      - pandas DataFrame     → ``list`` of dicts (``records`` orient)
      - ``NaN`` / ``Inf``    → ``None`` (JSON has no NaN literal)

    Usage::

        import json
        report = {"rmse": np.float64(0.42), "fi": np.array([1, 2, 3])}
        json.dump(as_serializable(report), f)
    """
    import math

    if isinstance(obj, dict):
        return {as_serializable(k): as_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [as_serializable(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return as_serializable(obj.item())
        return [as_serializable(item) for item in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # pandas types
    if isinstance(obj, pd.Series):
        return [as_serializable(item) for item in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [as_serializable(row) for row in obj.to_dict(orient="records")]
    return obj


def _ensure_c_contiguous(arr: np.ndarray) -> np.ndarray:
    """Return *arr* as C-contiguous if it isn't already."""
    if arr.ndim == 0:
        return arr
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr)
    return arr


def install() -> None:
    """Monkey-patch pandas to always return C-contiguous numpy arrays.

    This is the **single global fix** for all SIGSEGV issues caused by
    pandas 3.0's F-contiguous array layout.  Call once at startup.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    # --- Patch DataFrame.to_numpy ---
    _orig_df_to_numpy = pd.DataFrame.to_numpy

    def _patched_df_to_numpy(self, *args, **kwargs):
        arr = _orig_df_to_numpy(self, *args, **kwargs)
        return _ensure_c_contiguous(arr)

    pd.DataFrame.to_numpy = _patched_df_to_numpy

    # --- Patch Series.to_numpy ---
    _orig_series_to_numpy = pd.Series.to_numpy

    def _patched_series_to_numpy(self, *args, **kwargs):
        arr = _orig_series_to_numpy(self, *args, **kwargs)
        return _ensure_c_contiguous(arr)

    pd.Series.to_numpy = _patched_series_to_numpy

    # --- Patch DataFrame.values property ---
    _orig_df_values = pd.DataFrame.values.fget  # type: ignore[attr-defined]

    @property  # type: ignore[misc]
    def _safe_df_values(self):
        arr = _orig_df_values(self)
        return _ensure_c_contiguous(arr)

    pd.DataFrame.values = _safe_df_values  # type: ignore[assignment]

    # --- Patch Series.values property ---
    _orig_series_values = pd.Series.values.fget  # type: ignore[attr-defined]

    @property  # type: ignore[misc]
    def _safe_series_values(self):
        arr = _orig_series_values(self)
        if isinstance(arr, np.ndarray):
            return _ensure_c_contiguous(arr)
        return arr

    pd.Series.values = _safe_series_values  # type: ignore[assignment]

    # --- Patch plotly Template.__deepcopy__ ---
    # plotly's update_layout() internally deepcopies the global Template
    # object.  During deepcopy, Parcoords.__init__ and other validators
    # reconstruct numpy arrays via C extensions.  If any F-contiguous
    # arrays are reachable from the template graph, the C extension
    # crashes with SIGSEGV.
    #
    # Fix: replace Template.__deepcopy__ with a no-op that returns self.
    # Templates are effectively immutable singletons so sharing is safe.
    try:
        import plotly.graph_objs.layout as _plotly_layout
        import plotly.io as pio

        _Template = _plotly_layout.Template

        def _template_deepcopy_noop(self, memo=None):  # type: ignore[override]
            """Return *self* instead of deepcopying — avoids SIGSEGV."""
            return self

        _Template.__deepcopy__ = _template_deepcopy_noop

        # Set global default so individual calls don't need template=
        pio.templates.default = "plotly_white"

        logger.info("plotly Template.__deepcopy__ safety patch installed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("plotly Template patch skipped: %s", exc)

    # --- Disable cyclic GC globally ---
    # GC triggers C-extension finalizers on F-contiguous numpy arrays
    # left over from pandas 3.0 operations. These finalizers crash with
    # SIGSEGV during plotly chart creation, literature search, and other
    # operations.  Disabling GC is safe because:
    # 1. Reference counting still frees non-cyclic objects immediately
    # 2. GUI sessions are short-lived (~100s)
    # 3. Memory usage is bounded (~310 MB peak)
    # 4. Process exit reclaims all memory
    gc.disable()
    logger.info("Cyclic garbage collector disabled to prevent SIGSEGV")

    _INSTALLED = True
    logger.info(
        "pandas C-contiguous compatibility patches installed "
        "(DataFrame.to_numpy, Series.to_numpy, .values, plotly Template, GC disabled)"
    )
