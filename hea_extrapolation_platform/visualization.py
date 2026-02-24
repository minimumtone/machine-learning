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

from hea_extrapolation_platform.evaluation import ValidityScore
from hea_extrapolation_platform.ood import OODResult
from hea_extrapolation_platform.workflows import RunResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style: doubled font sizes for presentations
# ---------------------------------------------------------------------------
_BASE_FONT = 12
plt.rcParams.update({
    "font.size": _BASE_FONT * 2,
    "axes.titlesize": _BASE_FONT * 2.2,
    "axes.labelsize": _BASE_FONT * 2,
    "xtick.labelsize": _BASE_FONT * 1.6,
    "ytick.labelsize": _BASE_FONT * 1.6,
    "legend.fontsize": _BASE_FONT * 1.6,
    "figure.titlesize": _BASE_FONT * 2.4,
    "figure.dpi": 150,
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
    coords = pca.fit_transform(X_all.values)
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
    fig.savefig(fpath, bbox_inches="tight")
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
    coords = reducer.fit_transform(X_all.values)
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
    fig.savefig(fpath, bbox_inches="tight")
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
    left = np.zeros(len(fs_names))

    for dim, label, color in zip(dims, dim_labels, colors):
        vals = np.array([getattr(s, dim) for s in scores])
        if dim == "leak_penalty":
            vals = -vals  # show as negative contribution
        ax.barh(y_pos, vals, left=left, height=0.6, label=label, color=color)
        left += vals

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
    fig.savefig(fpath, bbox_inches="tight")
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
    df = pd.DataFrame(records)
    pivot = df.groupby(["feature_set", "split_policy"])[metric].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 1.0)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=_BASE_FONT * 1.4, color="black")

    ax.set_title(f"Performance Comparison ({metric})")
    ax.set_xlabel("Split Policy")
    ax.set_ylabel("Feature Set")

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight")
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
) -> Path:
    """Combined parity plot of y_true vs y_pred for all runs."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 10))

    all_true: List[float] = []
    all_pred: List[float] = []
    for r in runs:
        if r.y_test_true is not None and r.y_test_pred is not None:
            all_true.extend(r.y_test_true.tolist())
            all_pred.extend(r.y_test_pred.tolist())

    if all_true:
        ax.scatter(all_true, all_pred, alpha=0.15, s=30, color="#4C72B0")
        lo = min(min(all_true), min(all_pred))
        hi = max(max(all_true), max(all_pred))
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", linewidth=1, label="y = x")
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

    ax.set_xlabel("True Yield Strength (MPa)")
    ax.set_ylabel("Predicted Yield Strength (MPa)")
    ax.set_title(title)
    ax.legend()

    fpath = out_dir / filename
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight")
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
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved uncertainty-vs-OOD plot: %s", fpath)
    return fpath
