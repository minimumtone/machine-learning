"""
Visualization Module for HEA Extrapolation Platform
可視化モジュール

Generates publication-quality figures for:
  - OOD cluster maps (PCA / UMAP)
  - Feature validity ranking bar charts
  - Split-wise performance comparison tables
  - Prediction uncertainty scatter plots

All figures use doubled font sizes for presentation readability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from extrapolation_discovery_platform.evaluation import ValidityScore
from extrapolation_discovery_platform.ood import OODResult
from extrapolation_discovery_platform.workflows import RunResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style: doubled font sizes for presentations
# NOTE: figure.dpi is intentionally NOT set globally.  Setting a high dpi
#       (e.g. 150) in rcParams inflates the AGG raster buffer for *every*
#       figure (1500×1500 px @ figsize 10×10 with dpi=150).  Combined with
#       scatter plots of 100k+ points this exhausts memory and triggers a
#       SIGSEGV in the AGG C layer.  Each savefig() call below uses
#       dpi=_SAVE_DPI instead so the caller controls resolution explicitly.
# ---------------------------------------------------------------------------
_BASE_FONT = 12
_SAVE_DPI = 100  # safe default; increase to 150 only if post-processing needs it
plt.rcParams.update({
    "font.size": _BASE_FONT * 2,
    "axes.titlesize": _BASE_FONT * 2.2,
    "axes.labelsize": _BASE_FONT * 2,
    "xtick.labelsize": _BASE_FONT * 1.6,
    "ytick.labelsize": _BASE_FONT * 1.6,
    "legend.fontsize": _BASE_FONT * 1.6,
    "figure.titlesize": _BASE_FONT * 2.4,
    # figure.dpi deliberately omitted — controlled per savefig() call.
})


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. OOD cluster map (PCA)
# ---------------------------------------------------------------------------

def plot_ood_map_pca(
    X_train: pd.DataFrame,
    X_query: pd.DataFrame,
    ood_result: OODResult,
    out_dir: Path,
    filename: str = "ood_map_pca.png",
    title: str = "OOD Map (PCA)",
) -> Path:
    """2-D PCA projection of train + query coloured by OOD score.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_query : pd.DataFrame
        Query features (test / candidate).
    ood_result : OODResult
        OOD detection output for the query set.
    out_dir : Path
        Directory to save the figure.
    filename : str
        Output filename.
    title : str
        Figure title.

    Returns
    -------
    Path to the saved figure.
    """
    _ensure_dir(out_dir)

    pca = PCA(n_components=2)
    X_all = pd.concat([X_train, X_query], axis=0, ignore_index=True)
    # CRITICAL: Force C-contiguous layout.  pandas 3.0 DataFrame.values
    # returns F-contiguous arrays from fragmented BlockManagers, causing
    # SIGSEGV in BLAS/LAPACK calls inside PCA.
    X_arr = np.ascontiguousarray(
        X_all.to_numpy(dtype="float64", na_value=np.nan)
    )
    coords = pca.fit_transform(X_arr)
    n_train = len(X_train)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Training points (grey)
    ax.scatter(
        coords[:n_train, 0], coords[:n_train, 1],
        c="grey", alpha=0.3, s=40, label="Train",
    )

    # Query points coloured by composite OOD score
    sc = ax.scatter(
        coords[n_train:, 0], coords[n_train:, 1],
        c=ood_result.composite_scores,
        cmap="RdYlGn_r", s=80, edgecolors="k", linewidths=0.5,
        label="Query",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("OOD Score")

    # Mark OOD samples with red edge
    ood_idx = np.where(ood_result.is_ood)[0]
    if len(ood_idx) > 0:
        ax.scatter(
            coords[n_train + ood_idx, 0], coords[n_train + ood_idx, 1],
            facecolors="none", edgecolors="red", s=160, linewidths=2,
            label=f"OOD (n={len(ood_idx)})",
        )

    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight", dpi=_SAVE_DPI)
    plt.close(fig)
    logger.info("Saved OOD map: %s", fpath)
    return fpath


# ---------------------------------------------------------------------------
# 2. OOD map with UMAP (optional)
# ---------------------------------------------------------------------------

def plot_ood_map_umap(
    X_train: pd.DataFrame,
    X_query: pd.DataFrame,
    ood_result: OODResult,
    out_dir: Path,
    filename: str = "ood_map_umap.png",
    title: str = "OOD Map (UMAP)",
) -> Optional[Path]:
    """2-D UMAP projection. Returns None if umap-learn not installed."""
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed – skipping UMAP plot")
        return None

    _ensure_dir(out_dir)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
    X_all = pd.concat([X_train, X_query], axis=0, ignore_index=True)
    # CRITICAL: Force C-contiguous layout to avoid SIGSEGV
    X_arr = np.ascontiguousarray(
        X_all.to_numpy(dtype="float64", na_value=np.nan)
    )
    coords = reducer.fit_transform(X_arr)
    n_train = len(X_train)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        coords[:n_train, 0], coords[:n_train, 1],
        c="grey", alpha=0.3, s=40, label="Train",
    )
    sc = ax.scatter(
        coords[n_train:, 0], coords[n_train:, 1],
        c=ood_result.composite_scores,
        cmap="RdYlGn_r", s=80, edgecolors="k", linewidths=0.5,
    )
    fig.colorbar(sc, ax=ax, label="OOD Score")

    ood_idx = np.where(ood_result.is_ood)[0]
    if len(ood_idx) > 0:
        ax.scatter(
            coords[n_train + ood_idx, 0], coords[n_train + ood_idx, 1],
            facecolors="none", edgecolors="red", s=160, linewidths=2,
            label=f"OOD (n={len(ood_idx)})",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(loc="upper right")

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight", dpi=_SAVE_DPI)
    plt.close(fig)
    logger.info("Saved UMAP OOD map: %s", fpath)
    return fpath


# ---------------------------------------------------------------------------
# 3. Feature validity ranking bar chart
# ---------------------------------------------------------------------------

def plot_validity_ranking(
    scores: List[ValidityScore],
    out_dir: Path,
    filename: str = "validity_ranking.png",
) -> Path:
    """Stacked horizontal bar chart of validity scores per feature set."""
    _ensure_dir(out_dir)

    fs_names = [s.feature_set for s in scores]
    dims = ["effect_size", "stability", "generalisation", "leak_penalty", "extrapolation_safety"]
    dim_labels = ["Effect Size", "Stability", "Generalisation", "-Leak Penalty", "Extrap. Safety"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8C8C8C", "#CCB974"]

    fig, ax = plt.subplots(figsize=(14, max(6, len(fs_names) * 1.2)))

    y_pos = np.arange(len(fs_names))
    left_pos = np.zeros(len(fs_names))  # accumulator for positive bars
    left_neg = np.zeros(len(fs_names))  # accumulator for negative bars

    for dim, label, color in zip(dims, dim_labels, colors):
        vals = np.array([getattr(s, dim) for s in scores])
        if dim == "leak_penalty":
            # Show leak penalty as a separate negative bar from zero,
            # so it does not shift the positive bar stack.
            ax.barh(y_pos, -vals, left=left_neg, height=0.6, label=label, color=color)
            left_neg -= vals
        else:
            ax.barh(y_pos, vals, left=left_pos, height=0.6, label=label, color=color)
            left_pos += vals

    # Overlay total score as marker
    totals = [s.total for s in scores]
    ax.scatter(totals, y_pos, color="black", zorder=5, s=100, marker="D", label="Total")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fs_names)
    ax.set_xlabel("Score")
    ax.set_title("Feature Set Validity Ranking")
    ax.legend(loc="lower right", fontsize=_BASE_FONT * 1.2)
    ax.axvline(0, color="black", linewidth=0.5)

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight", dpi=_SAVE_DPI)
    plt.close(fig)
    logger.info("Saved validity ranking: %s", fpath)
    return fpath


# ---------------------------------------------------------------------------
# 4. Split-wise performance comparison heatmap
# ---------------------------------------------------------------------------

def plot_performance_heatmap(
    runs: List[RunResult],
    out_dir: Path,
    filename: str = "performance_heatmap.png",
    metric: str = "rmse_test",
) -> Path:
    """Heatmap of metric (feature_set x split_policy), averaged across seeds/folds."""
    _ensure_dir(out_dir)

    records = []
    for r in runs:
        records.append({
            "feature_set": r.feature_set,
            "split_policy": r.split_policy,
            metric: getattr(r, metric),
        })
    # Columnar construction to avoid DataFrame fragmentation / SIGSEGV
    if records:
        col_names = list(records[0].keys())
        df = pd.DataFrame({k: [rec[k] for rec in records] for k in col_names})
    else:
        df = pd.DataFrame()
    pivot = df.groupby(["feature_set", "split_policy"])[metric].mean().unstack(fill_value=0)
    # Force C-contiguous for pivot array used in imshow
    pivot_arr = np.ascontiguousarray(
        pivot.to_numpy(dtype="float64", na_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 1.0)))
    im = ax.imshow(pivot_arr, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot_arr[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=_BASE_FONT * 1.4, color="black")

    ax.set_title(f"Performance Comparison ({metric})")
    ax.set_xlabel("Split Policy")
    ax.set_ylabel("Feature Set")

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight", dpi=_SAVE_DPI)
    plt.close(fig)
    logger.info("Saved performance heatmap: %s", fpath)
    return fpath


# ---------------------------------------------------------------------------
# 5. Prediction vs True scatter (with uncertainty)
# ---------------------------------------------------------------------------

def plot_parity(
    runs: List[RunResult],
    out_dir: Path,
    filename: str = "parity_plot.png",
    title: str = "Parity Plot (Test Set)",
    max_points: int = 20_000,
) -> Path:
    """Combined parity plot of y_true vs y_pred for all runs.

    Parameters
    ----------
    max_points : int
        Maximum number of points to render.  When the combined run set
        exceeds this, a random subsample is drawn (reproducible seed=0).
        This prevents AGG buffer overflows that cause SIGSEGV when
        rendering hundreds of thousands of semi-transparent markers.
    """
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 10))

    all_true: List[float] = []
    all_pred: List[float] = []
    for r in runs:
        if r.y_test_true is not None and r.y_test_pred is not None:
            all_true.extend(r.y_test_true.tolist())
            all_pred.extend(r.y_test_pred.tolist())

    if all_true:
        arr_true = np.asarray(all_true, dtype=np.float64)
        arr_pred = np.asarray(all_pred, dtype=np.float64)

        # Remove NaN / Inf so axis limits and scatter are well-defined.
        finite_mask = np.isfinite(arr_true) & np.isfinite(arr_pred)
        n_removed = int((~finite_mask).sum())
        if n_removed > 0:
            logger.warning(
                "plot_parity: removed %d non-finite (NaN/Inf) points "
                "from %d total", n_removed, len(arr_true),
            )
        arr_true = arr_true[finite_mask]
        arr_pred = arr_pred[finite_mask]

        # Subsample to avoid AGG SIGSEGV with large point clouds.
        n_pts = len(arr_true)
        if n_pts > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_pts, size=max_points, replace=False)
            arr_true = arr_true[idx]
            arr_pred = arr_pred[idx]
            logger.info(
                "plot_parity: subsampled %d → %d points for rendering",
                n_pts, max_points,
            )

        if len(arr_true) > 0:
            # rasterized=True offloads marker rendering to the AGG raster
            # layer as a single image, avoiding per-marker path overhead and
            # the associated C-level buffer that can overflow.
            ax.scatter(
                arr_true, arr_pred,
                alpha=0.15, s=30, color="#4C72B0",
                rasterized=True,
            )
            lo = float(min(arr_true.min(), arr_pred.min()))
            hi = float(max(arr_true.max(), arr_pred.max()))
            margin = (hi - lo) * 0.05
            ax.plot(
                [lo - margin, hi + margin],
                [lo - margin, hi + margin],
                "k--", linewidth=1, label="y = x",
            )
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

    ax.set_xlabel("True Yield Strength (MPa)")
    ax.set_ylabel("Predicted Yield Strength (MPa)")
    ax.set_title(title)
    ax.legend()

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight", dpi=_SAVE_DPI)
    plt.close(fig)
    logger.info("Saved parity plot: %s", fpath)
    return fpath


# ---------------------------------------------------------------------------
# 6. Uncertainty vs OOD score scatter
# ---------------------------------------------------------------------------

def plot_uncertainty_vs_ood(
    ood_scores: np.ndarray,
    uncertainties: np.ndarray,
    errors: np.ndarray,
    out_dir: Path,
    filename: str = "uncertainty_vs_ood.png",
) -> Path:
    """Scatter: x = OOD score, y = prediction uncertainty, colour = |error|."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(
        ood_scores, uncertainties,
        c=np.abs(errors), cmap="hot", s=60, edgecolors="k", linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("|Prediction Error| (MPa)")

    ax.set_xlabel("OOD Score")
    ax.set_ylabel("Prediction Uncertainty (std)")
    ax.set_title("Uncertainty vs OOD Score")

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight", dpi=_SAVE_DPI)
    plt.close(fig)
    logger.info("Saved uncertainty-vs-OOD plot: %s", fpath)
    return fpath
