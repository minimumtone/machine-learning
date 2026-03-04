"""
Multicollinearity Diagnostics & Model Selection Module
多重共線性診断・モデル自動選択モジュール

Phase 0 of the experiment pipeline: diagnose multicollinearity per feature set,
remove constant / perfectly collinear columns, compute VIF, and automatically
select which ML workflows are appropriate for each feature set.

Design:
  - All functions accept/return plain Python types or C-contiguous numpy arrays
    to prevent SIGSEGV with pandas 3.0 F-contiguous layouts.
  - The module is imported by runner.py; it has no dependency on GUI or plotly.
  - All arguments are Optional where possible to preserve backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — model selection thresholds
# ---------------------------------------------------------------------------

VIF_HIGH_THRESHOLD = 10.0       # VIF > this → 'high multicollinearity'
VIF_MODERATE_THRESHOLD = 5.0    # VIF > this → 'moderate'
HIGH_VIF_RATIO_CUTOFF = 0.30    # high VIF feature ratio above this → block linear models
MAGPIE_DIM_THRESHOLD = 100      # n_features >= this → 'high-dimensional'

# Priority columns to KEEP when breaking collinear ties
# (physically interpretable HEA descriptors preferred over derived statistics)
PRIORITY_KEEP = [
    'VEC', 'delta_r', 'dS_mix', 'dH_mix', 'delta_EN',
    'Tm_avg', 'omega', 'elastic_mismatch',
]


# ---------------------------------------------------------------------------
# 3.1.1  compute_vif() — VIF calculation
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R^2_j) where R^2_j is from regressing X_j on all others.

    Special cases:
      - Constant columns (std ~0) -> VIF = inf
      - Near-singular design -> VIF = inf

    Returns pd.Series sorted descending by VIF.
    """
    from sklearn.linear_model import LinearRegression

    result: Dict[str, float] = {}
    cols = list(X.columns)
    # Ensure C-contiguous float64 array for BLAS safety
    X_arr = np.ascontiguousarray(X.to_numpy(dtype='float64', na_value=np.nan))

    for i, col in enumerate(cols):
        y = X_arr[:, i].copy()
        if np.std(y) < 1e-10:
            result[col] = float('inf')  # constant column
            continue
        others = np.ascontiguousarray(np.delete(X_arr, i, axis=1))
        lr = LinearRegression(fit_intercept=True)
        lr.fit(others, y)
        r2 = float(lr.score(others, y))
        r2 = min(r2, 1.0 - 1e-10)  # numerical guard
        result[col] = 1.0 / (1.0 - r2)

    return pd.Series(result).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 3.1.2  remove_perfect_collinear() — perfect collinearity removal
# ---------------------------------------------------------------------------

def remove_perfect_collinear(
    X: pd.DataFrame,
    threshold: float = 0.9999,
    priority: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove one column from each near-perfectly-collinear pair.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    threshold : float
        Absolute correlation threshold for "perfect" collinearity.
    priority : list of str, optional
        Preferred columns to KEEP (e.g. physically interpretable features).

    Returns
    -------
    (filtered_df, list_of_dropped_columns)
    """
    dropped: List[str] = []
    df = X.copy()
    priority_set = set(priority) if priority else set()

    while True:
        if df.shape[1] < 2:
            break
        # Use C-contiguous values for corr computation
        corr = df.corr().abs()
        # pandas 3.0: .values / to_numpy() may return read-only array;
        # np.array(..., copy=True) guarantees a writable, C-contiguous copy
        corr_arr = np.array(corr.to_numpy(dtype='float64'), order='C', copy=True)
        np.fill_diagonal(corr_arr, 0.0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        max_corr = float(corr.max().max())
        if max_corr < threshold:
            break
        # Find the pair
        col_a, col_b = corr.stack().idxmax()
        # Drop: prefer to keep the one in priority
        if priority_set:
            a_in = col_a in priority_set
            b_in = col_b in priority_set
            if a_in and not b_in:
                to_drop = col_b
            elif b_in and not a_in:
                to_drop = col_a
            else:
                to_drop = col_b  # both or neither in priority → drop second
        else:
            to_drop = col_b
        keep = col_a if to_drop == col_b else col_b
        df = df.drop(columns=[to_drop])
        dropped.append(to_drop)
        logger.warning(
            'Removed perfect-collinear column: %s (|r|=%.4f with %s)',
            to_drop, max_corr, keep,
        )
    return df, dropped


# ---------------------------------------------------------------------------
# 3.1.3  remove_constant_columns() — constant column removal
# ---------------------------------------------------------------------------

def remove_constant_columns(
    X: pd.DataFrame,
    variance_threshold: float = 1e-10,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns with near-zero variance (constants)."""
    stds = X.std()
    const_cols = stds[stds < variance_threshold].index.tolist()
    if const_cols:
        logger.warning('Removing %d constant columns: %s', len(const_cols), const_cols)
    return X.drop(columns=const_cols), const_cols


# ---------------------------------------------------------------------------
# 3.1.4  MulticollinearityReport — diagnostic report dataclass
# ---------------------------------------------------------------------------

@dataclass
class MulticollinearityReport:
    """Diagnostic result for one feature set."""

    feature_set: str
    n_features_before: int
    n_features_after: int
    dropped_constant: List[str]
    dropped_perfect: List[str]
    vif_series: pd.Series
    high_vif_count: int
    moderate_vif_count: int
    multicollinearity_level: str   # 'low' | 'moderate' | 'high'
    recommended_workflows: List[str]
    blocked_workflows: List[str]

    @property
    def high_vif_ratio(self) -> float:
        return self.high_vif_count / max(self.n_features_after, 1)

    def to_dict(self) -> dict:
        """Serialise for logging / JSON export (exclude pd.Series for safety)."""
        return {
            'feature_set': self.feature_set,
            'n_features_before': self.n_features_before,
            'n_features_after': self.n_features_after,
            'dropped_constant': self.dropped_constant,
            'dropped_perfect': self.dropped_perfect,
            'high_vif_count': self.high_vif_count,
            'moderate_vif_count': self.moderate_vif_count,
            'multicollinearity_level': self.multicollinearity_level,
            'high_vif_ratio': round(self.high_vif_ratio, 4),
            'recommended_workflows': self.recommended_workflows,
            'blocked_workflows': self.blocked_workflows,
        }


# ---------------------------------------------------------------------------
# 4.1  select_workflows_for_feature_set()
# ---------------------------------------------------------------------------

def select_workflows_for_feature_set(
    report: MulticollinearityReport,
    all_workflows: List[str],
    n_samples: int,
) -> Tuple[List[str], List[str], str]:
    """Select which workflows are appropriate for a feature set.

    Returns (allowed_workflows, blocked_workflows, selection_reason).

    Selection logic (evaluated in order):
    1. High-dim guard: n_features >= MAGPIE_DIM_THRESHOLD and n < 400
         → Block WF-LIN, WF-ARD
    2. High VIF guard: high_vif_ratio > HIGH_VIF_RATIO_CUTOFF
         → Block WF-LIN, WF-LASSO, WF-ARD
    3. Moderate VIF: moderate ratio > 0.30
         → WF-LIN requires PCA (signalled via reason string)
    4. Otherwise: allow all
    """
    blocked: set = set()
    reason_parts: List[str] = []

    # 1. High-dimensionality guard
    if report.n_features_after >= MAGPIE_DIM_THRESHOLD and n_samples < 400:
        blocked.update(['WF-LIN', 'WF-ARD'])
        reason_parts.append(
            f'High-dim ({report.n_features_after}D / n={n_samples}): '
            f'WF-LIN, WF-ARD blocked'
        )

    # 2. High VIF guard
    if report.high_vif_ratio > HIGH_VIF_RATIO_CUTOFF:
        blocked.update(['WF-LIN', 'WF-LASSO', 'WF-ARD'])
        reason_parts.append(
            f'{report.high_vif_count} features VIF>10 '
            f'({report.high_vif_ratio:.0%}): linear models blocked'
        )

    # 3. Moderate VIF — force PCA on for WF-LIN
    moderate_ratio = report.moderate_vif_count / max(report.n_features_after, 1)
    if moderate_ratio > 0.30 and 'WF-LIN' not in blocked:
        reason_parts.append(
            f'{report.moderate_vif_count} features VIF>5: WF-LIN requires PCA'
        )

    allowed = [w for w in all_workflows if w not in blocked]
    blocked_list = sorted(blocked)
    reason = '; '.join(reason_parts) if reason_parts else 'no multicollinearity restriction'
    logger.info(
        'Model selection [%s]: allowed=%s blocked=%s | %s',
        report.feature_set, allowed, blocked_list, reason,
    )
    return allowed, blocked_list, reason


# ---------------------------------------------------------------------------
# 4.2  run_phase0_multicollinearity() — Phase 0 integration function
# ---------------------------------------------------------------------------

def run_phase0_multicollinearity(
    features_all: pd.DataFrame,
    feature_sets: List[FeatureSetName],
    all_workflows: List[str],
    n_samples: int,
) -> Dict[str, MulticollinearityReport]:
    """Run full multicollinearity pipeline for every feature set.

    Called by runner.py before Phase 1. Returns {fs_name: MulticollinearityReport}.
    """
    reports: Dict[str, MulticollinearityReport] = {}

    for fs in feature_sets:
        fs_key = fs.value
        cols = FeatureCatalog.columns(fs)
        # Ensure only columns present in features_all are used
        available_cols = [c for c in cols if c in features_all.columns]
        if not available_cols:
            logger.warning('Phase 0: no columns available for %s, skipping', fs_key)
            continue

        X_fs = features_all[available_cols].copy()
        n_before = X_fs.shape[1]

        # Step A: Remove constant columns
        X_fs, dropped_const = remove_constant_columns(X_fs)

        # Step B: Remove perfect collinear columns
        X_fs, dropped_perfect = remove_perfect_collinear(
            X_fs, threshold=0.9999, priority=PRIORITY_KEEP
        )

        # Step C: Compute VIF on cleaned features
        if X_fs.shape[1] > 1:
            vif = compute_vif(X_fs)
        else:
            # Single feature — VIF is trivially 1.0
            vif = pd.Series(
                {X_fs.columns[0]: 1.0} if X_fs.shape[1] == 1 else {}
            )

        high_count = int((vif > VIF_HIGH_THRESHOLD).sum())
        mod_count = int(
            ((vif > VIF_MODERATE_THRESHOLD) & (vif <= VIF_HIGH_THRESHOLD)).sum()
        )
        high_ratio = high_count / max(len(vif), 1)

        if high_ratio > HIGH_VIF_RATIO_CUTOFF:
            level = 'high'
        elif (high_count + mod_count) / max(len(vif), 1) > 0.30:
            level = 'moderate'
        else:
            level = 'low'

        # Step D: Model selection
        report = MulticollinearityReport(
            feature_set=fs_key,
            n_features_before=n_before,
            n_features_after=len(vif),
            dropped_constant=dropped_const,
            dropped_perfect=dropped_perfect,
            vif_series=vif,
            high_vif_count=high_count,
            moderate_vif_count=mod_count,
            multicollinearity_level=level,
            recommended_workflows=[],
            blocked_workflows=[],
        )
        allowed, blocked, reason = select_workflows_for_feature_set(
            report, all_workflows, n_samples
        )
        report.recommended_workflows = allowed
        report.blocked_workflows = blocked
        reports[fs_key] = report

        logger.info(
            'Phase 0 [%s]: %dD→%dD (dropped const=%d perfect=%d) '
            'VIF_high=%d VIF_mod=%d level=%s | %s',
            fs_key, n_before, len(vif),
            len(dropped_const), len(dropped_perfect),
            high_count, mod_count, level, reason,
        )

    return reports
