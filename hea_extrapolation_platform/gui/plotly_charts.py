"""
Plotly Interactive Charts for Extrapolation Discovery Platform
Plotlyインタラクティブチャートモジュール

Provides interactive equivalents of the matplotlib-based visualization module:
  - plotly_ood_map        : OOD cluster map with hover + lasso select
  - plotly_validity_ranking : Feature validity stacked bar chart
  - plotly_heatmap        : Performance heatmap (feature_set × split_policy)
  - plotly_parity         : Parity plot (y_true vs y_pred)
  - plotly_uncertainty_ood : Uncertainty vs OOD score scatter
  - plotly_feature_frequency : Literature feature frequency bar chart
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from a list-of-dicts using columnar construction.

    ``pd.DataFrame(list_of_dicts)`` creates one internal memory block per
    dict key, leading to a fragmented BlockManager.  Downstream numpy
    interop (``.describe()``, ``.corr()``, Plotly charting) on such a
    frame can trigger a SIGSEGV in the pandas/numpy C layer.

    Building from ``{col: [values]}`` creates a single consolidated block.
    """
    if not records:
        return pd.DataFrame()
    col_names = list(records[0].keys())
    return pd.DataFrame({k: [r[k] for r in records] for k in col_names})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. OOD Cluster Map (PCA)
# ---------------------------------------------------------------------------

def plotly_ood_map(
    X_train: pd.DataFrame,
    X_query: pd.DataFrame,
    composite_scores: np.ndarray,
    is_ood: np.ndarray,
    title: str = "OOD Map (PCA)",
) -> go.Figure:
    """Interactive 2-D PCA projection coloured by OOD score.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_query : pd.DataFrame
        Query feature matrix.
    composite_scores : np.ndarray
        OOD composite score per query sample.
    is_ood : np.ndarray
        Boolean mask — True if sample is OOD.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    X_all = pd.concat([X_train, X_query], axis=0, ignore_index=True)
    # CRITICAL: Force C-contiguous layout before StandardScaler / PCA.
    # pandas 3.0 DataFrame.values returns F-contiguous (column-major) arrays
    # when the BlockManager is fragmented.  BLAS/LAPACK calls inside
    # StandardScaler and PCA assume C-contiguous and SIGSEGV otherwise.
    X_arr = np.ascontiguousarray(
        X_all.to_numpy(dtype="float64", na_value=np.nan)
    )
    scaler = StandardScaler()
    X_scaled = np.ascontiguousarray(scaler.fit_transform(X_arr))
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    n_train = len(X_train)
    var_ratio = pca.explained_variance_ratio_

    fig = go.Figure()

    # Training points
    fig.add_trace(go.Scatter(
        x=coords[:n_train, 0],
        y=coords[:n_train, 1],
        mode="markers",
        marker=dict(color="grey", opacity=0.3, size=6),
        name="Train",
        hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra>Train</extra>",
    ))

    # Query points — coloured by OOD score
    query_x = coords[n_train:, 0]
    query_y = coords[n_train:, 1]
    in_dist_mask = ~is_ood

    # In-distribution query points
    if in_dist_mask.any():
        fig.add_trace(go.Scatter(
            x=query_x[in_dist_mask],
            y=query_y[in_dist_mask],
            mode="markers",
            marker=dict(
                color=composite_scores[in_dist_mask],
                colorscale="RdYlGn_r",
                size=8,
                colorbar=dict(title="OOD Score"),
                line=dict(width=0.5, color="black"),
            ),
            name="Query (In-Dist)",
            hovertemplate=(
                "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>"
                "OOD Score: %{marker.color:.3f}<extra>In-Dist</extra>"
            ),
        ))

    # OOD query points — red ring markers
    if is_ood.any():
        fig.add_trace(go.Scatter(
            x=query_x[is_ood],
            y=query_y[is_ood],
            mode="markers",
            marker=dict(
                color=composite_scores[is_ood],
                colorscale="RdYlGn_r",
                size=12,
                line=dict(width=2, color="red"),
            ),
            name=f"OOD (n={int(is_ood.sum())})",
            hovertemplate=(
                "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>"
                "OOD Score: %{marker.color:.3f}<extra>OOD</extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 — 第1主成分 ({var_ratio[0]*100:.1f}% 分散説明)",
        yaxis_title=f"PC2 — 第2主成分 ({var_ratio[1]*100:.1f}% 分散説明)",
        template="plotly_white",
        dragmode="lasso",
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Feature Validity Ranking (Stacked Bar)
# ---------------------------------------------------------------------------

def plotly_validity_ranking(
    scores: List[Any],
) -> go.Figure:
    """Interactive stacked horizontal bar chart of validity scores.

    Parameters
    ----------
    scores : list of ValidityScore
        Sorted by total score (descending).

    Returns
    -------
    go.Figure
    """
    fs_names = [s.feature_set for s in scores]
    dims = [
        ("effect_size", "Effect Size", "#4C72B0"),
        ("stability", "Stability", "#55A868"),
        ("generalisation", "Generalisation", "#C44E52"),
        ("extrapolation_safety", "Extrap. Safety", "#CCB974"),
    ]

    fig = go.Figure()

    for attr, label, color in dims:
        vals = [getattr(s, attr) for s in scores]
        fig.add_trace(go.Bar(
            y=fs_names,
            x=vals,
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"{label}: %{{x:.3f}}<extra>%{{y}}</extra>",
        ))

    # Leak penalty as negative bar
    leak_vals = [-s.leak_penalty for s in scores]
    fig.add_trace(go.Bar(
        y=fs_names,
        x=leak_vals,
        name="-Leak Penalty",
        orientation="h",
        marker_color="#8C8C8C",
        hovertemplate="-Leak: %{x:.3f}<extra>%{y}</extra>",
    ))

    # Total score overlay
    totals = [s.total for s in scores]
    fig.add_trace(go.Scatter(
        y=fs_names,
        x=totals,
        mode="markers",
        marker=dict(symbol="diamond", size=12, color="black"),
        name="Total",
        hovertemplate="Total: %{x:.3f}<extra>%{y}</extra>",
    ))

    fig.update_layout(
        barmode="relative",
        title="Feature Set Validity Ranking — 特徴量セット妥当性ランキング",
        xaxis_title="妥当性スコア (Score) — 高いほど良い",
        yaxis_title="特徴量セット (Feature Set)",
        template="plotly_white",
        height=max(400, len(fs_names) * 80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Performance Heatmap
# ---------------------------------------------------------------------------

def plotly_heatmap(
    runs: List[Any],
    metric: str = "rmse_test",
) -> go.Figure:
    """Interactive heatmap of metric (feature_set × split_policy).

    Parameters
    ----------
    runs : list of RunResult
    metric : str
        Metric attribute name (default 'rmse_test').

    Returns
    -------
    go.Figure
    """
    records = []
    for r in runs:
        records.append({
            "feature_set": r.feature_set,
            "split_policy": r.split_policy,
            metric: getattr(r, metric, 0.0),
        })
    df = _records_to_df(records)
    pivot = df.groupby(["feature_set", "split_policy"])[metric].mean().unstack(fill_value=0)
    # Force C-contiguous for pivot.values used by Plotly/numpy
    pivot_arr = np.ascontiguousarray(
        pivot.to_numpy(dtype="float64", na_value=0.0)
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_arr,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="YlOrRd",
        text=np.round(pivot_arr, 2),
        texttemplate="%{text}",
        hovertemplate=(
            "Feature Set: %{y}<br>Split: %{x}<br>"
            f"{metric}: %{{z:.2f}}<extra></extra>"
        ),
        colorbar=dict(title=metric.upper()),
    ))

    # Human-readable metric labels
    _metric_labels = {
        "rmse_test": "RMSE (Test) — 小さいほど良い",
        "rmse_train": "RMSE (Train) — 小さいほど良い",
        "mae_test": "MAE (Test) — 小さいほど良い",
        "mae_train": "MAE (Train) — 小さいほど良い",
        "r2_test": "R\u00b2 (Test) — 1に近いほど良い",
        "r2_train": "R\u00b2 (Train) — 1に近いほど良い",
    }
    metric_display = _metric_labels.get(metric, metric.upper())
    fig.update_layout(
        title=f"パフォーマンスヒートマップ: {metric_display}",
        xaxis_title="分割方法 (Split Policy)",
        yaxis_title="特徴量セット (Feature Set)",
        template="plotly_white",
        height=max(400, len(pivot) * 60 + 200),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Parity Plot
# ---------------------------------------------------------------------------

def plotly_parity(
    runs: List[Any],
    title: str = "Parity Plot (Test Set)",
) -> go.Figure:
    """Interactive parity plot of y_true vs y_pred for all runs.

    Parameters
    ----------
    runs : list of RunResult
    title : str

    Returns
    -------
    go.Figure
    """
    all_true: List[float] = []
    all_pred: List[float] = []
    all_wf: List[str] = []
    all_fs: List[str] = []

    # Fix #10: de-duplicate by (workflow, feature_set, sample_index)
    # When multiple seeds / folds produce predictions for the same sample,
    # keep only the first occurrence to avoid misleading scatter density.
    seen_keys: set = set()

    for r in runs:
        if r.y_test_true is not None and r.y_test_pred is not None:
            test_indices = getattr(r, "test_indices", None)
            for i in range(len(r.y_test_true)):
                if test_indices is not None:
                    key = (r.workflow, r.feature_set, int(test_indices[i]))
                else:
                    # Fallback: use (wf, fs, true_value) as rough key
                    key = (r.workflow, r.feature_set, float(r.y_test_true[i]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_true.append(float(r.y_test_true[i]))
                all_pred.append(float(r.y_test_pred[i]))
                all_wf.append(r.workflow)
                all_fs.append(r.feature_set)

    fig = go.Figure()

    if all_true:
        fig.add_trace(go.Scatter(
            x=all_true,
            y=all_pred,
            mode="markers",
            marker=dict(color="#4C72B0", opacity=0.2, size=5),
            customdata=list(zip(all_wf, all_fs)),
            hovertemplate=(
                "True: %{x:.1f}<br>Pred: %{y:.1f}<br>"
                "WF: %{customdata[0]}<br>FS: %{customdata[1]}"
                "<extra></extra>"
            ),
            name="Predictions",
        ))

        lo = min(min(all_true), min(all_pred))
        hi = max(max(all_true), max(all_pred))
        margin = (hi - lo) * 0.05
        fig.add_trace(go.Scatter(
            x=[lo - margin, hi + margin],
            y=[lo - margin, hi + margin],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="y = x",
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="実測値 (True Value)",
        yaxis_title="予測値 (Predicted Value)",
        template="plotly_white",
        height=600,
        width=600,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Uncertainty vs OOD Score
# ---------------------------------------------------------------------------

def plotly_uncertainty_ood(
    ood_scores: np.ndarray,
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> go.Figure:
    """Interactive scatter: x = OOD score, y = uncertainty, colour = |error|.

    Parameters
    ----------
    ood_scores : np.ndarray
    uncertainties : np.ndarray
    errors : np.ndarray

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(data=go.Scatter(
        x=ood_scores,
        y=uncertainties,
        mode="markers",
        marker=dict(
            color=np.abs(errors),
            colorscale="Hot",
            size=8,
            colorbar=dict(title="|Error|"),
            line=dict(width=0.3, color="black"),
        ),
        hovertemplate=(
            "OOD Score: %{x:.3f}<br>"
            "Uncertainty: %{y:.3f}<br>"
            "|Error|: %{marker.color:.1f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="予測不確実性 vs OODスコア",
        xaxis_title="OODスコア — 高いほど分布外",
        yaxis_title="予測不確実性 (std) — 高いほどばらつきが大きい",
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Literature Feature Frequency
# ---------------------------------------------------------------------------

def plotly_feature_frequency(
    feature_counts: List[Tuple[str, int]],
    title: str = "Feature Frequency in Literature",
) -> go.Figure:
    """Bar chart of feature frequency from literature search.

    Parameters
    ----------
    feature_counts : list of (feature_name, count)
    title : str

    Returns
    -------
    go.Figure
    """
    if not feature_counts:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[
            dict(text="No data", xref="paper", yref="paper",
                 x=0.5, y=0.5, showarrow=False, font_size=20)
        ])
        return fig

    names = [fc[0] for fc in feature_counts]
    counts = [fc[1] for fc in feature_counts]

    fig = go.Figure(data=go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker_color="#4C72B0",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="出現論文数 (Count) — 多いほど有効性が高い",
        yaxis_title="記述子 (Feature)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=max(300, len(names) * 30 + 100),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Run Results Table (DataFrame for display)
# ---------------------------------------------------------------------------

def runs_to_dataframe(runs: List[Any]) -> pd.DataFrame:
    """Convert list of RunResult to a summary DataFrame."""
    records = []
    for r in runs:
        records.append({
            "Workflow": r.workflow,
            "Feature Set": r.feature_set,
            "Split Policy": r.split_policy,
            "Seed": r.seed,
            "Fold": r.fold,
            "RMSE (Train)": round(r.rmse_train, 2),
            "RMSE (Test)": round(r.rmse_test, 2),
            "MAE (Train)": round(r.mae_train, 2),
            "MAE (Test)": round(r.mae_test, 2),
            "R² (Train)": round(r.r2_train, 4),
            "R² (Test)": round(r.r2_test, 4),
            "Time (s)": round(r.elapsed_sec, 2),
        })
    return _records_to_df(records)


def validity_scores_to_dataframe(scores: List[Any]) -> pd.DataFrame:
    """Convert list of ValidityScore to a summary DataFrame."""
    records = []
    for i, s in enumerate(scores):
        records.append({
            "Rank": i + 1,
            "Feature Set": s.feature_set,
            "Effect Size": round(s.effect_size, 4),
            "Stability": round(s.stability, 4),
            "Generalisation": round(s.generalisation, 4),
            "Leak Penalty": round(s.leak_penalty, 4),
            "Extrap. Safety": round(s.extrapolation_safety, 4),
            "Total": round(s.total, 4),
        })
    return _records_to_df(records)


# ---------------------------------------------------------------------------
# 8. Dataset Summary Statistics
# ---------------------------------------------------------------------------

def plotly_target_histogram(
    target: pd.Series,
    title: str = "Target Distribution",
) -> go.Figure:
    """Histogram of the target variable.

    Parameters
    ----------
    target : pd.Series
        Target values (e.g. yield strength).
    title : str

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(data=go.Histogram(
        x=np.ascontiguousarray(target.to_numpy(dtype="float64")),
        nbinsx=30,
        marker_color="#4C72B0",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=target.name or "Value",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )
    return fig


def plotly_composition_heatmap(
    compositions_df: pd.DataFrame,
    title: str = "Element Composition Heatmap",
) -> go.Figure:
    """Heatmap showing element fractions across all samples.

    Parameters
    ----------
    compositions_df : pd.DataFrame
        Composition table (element columns, fraction values).
    title : str

    Returns
    -------
    go.Figure
    """
    # Only show elements that are actually present (non-zero)
    non_zero_cols = [
        c for c in compositions_df.columns
        if compositions_df[c].sum() > 0
    ]
    if not non_zero_cols:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[
            dict(text="No data", xref="paper", yref="paper",
                 x=0.5, y=0.5, showarrow=False, font_size=20)
        ])
        return fig

    # Show mean composition per element as a bar chart
    means = compositions_df[non_zero_cols].mean().sort_values(ascending=False)
    stds = compositions_df[non_zero_cols].std()

    fig = go.Figure(data=go.Bar(
        x=list(means.index),
        y=list(means.to_numpy()),
        error_y=dict(
            type="data",
            array=[stds[c] for c in means.index],
            visible=True,
        ),
        marker_color="#55A868",
        hovertemplate="Element: %{x}<br>Mean: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Element",
        yaxis_title="Mean Atomic Fraction",
        template="plotly_white",
        height=350,
    )
    return fig


def plotly_feature_correlation(
    features_df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    max_features: int = 8,
    title: str = "Feature Correlation & Pair-Plot Matrix",
) -> go.Figure:
    """Pair-plot style correlation matrix.

    Layout:
      - **Lower triangle**: 2-D density contour (memory-efficient)
      - **Diagonal**: Histogram of each variable
      - **Upper triangle**: Correlation coefficient (text annotation)

    The *target* variable (if provided) is appended as the first
    column so users can assess feature--target relationships.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    target : pd.Series or None
        Target variable to include in the matrix.
    max_features : int
        Maximum number of features to show (picks highest-variance).
    title : str

    Returns
    -------
    go.Figure
    """
    from plotly.subplots import make_subplots

    if features_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Build combined DataFrame: target + top-variance features.
    # Avoid df.insert() which fragments the BlockManager; instead
    # build the final DataFrame in one shot via pd.concat.
    variances = features_df.var().sort_values(ascending=False)
    selected = list(variances.head(max_features).index)

    if target is not None and len(target) == len(features_df):
        tname = target.name if target.name else "Target"
        df = pd.concat(
            [target.rename(tname).reset_index(drop=True),
             features_df[selected].reset_index(drop=True)],
            axis=1,
        )
        cols = [tname] + selected
    else:
        df = features_df[selected].copy()
        cols = selected

    n = len(cols)
    if n < 2:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Consolidate to single C-contiguous memory block before .corr()
    # to prevent SIGSEGV in numpy C layer on fragmented DataFrames.
    try:
        _arr = np.ascontiguousarray(
            df[cols].to_numpy(dtype="float64", na_value=np.nan)
        )
        _tmp = pd.DataFrame(_arr, columns=cols, index=df.index)
    except (ValueError, TypeError):
        _tmp = df[cols]
    corr = _tmp.corr()

    fig = make_subplots(
        rows=n, cols=n,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Colour palette for correlation text
    def _corr_color(r: float) -> str:
        if abs(r) >= 0.7:
            return "#d32f2f" if r > 0 else "#1565c0"
        if abs(r) >= 0.4:
            return "#e65100" if r > 0 else "#0277bd"
        return "#616161"

    def _corr_size(r: float) -> int:
        if abs(r) >= 0.7:
            return 22
        if abs(r) >= 0.4:
            return 18
        return 14

    for i in range(n):
        for j in range(n):
            row = i + 1
            col = j + 1
            xi = np.ascontiguousarray(df[cols[j]].to_numpy(dtype="float64"))
            yi = np.ascontiguousarray(df[cols[i]].to_numpy(dtype="float64"))

            if i == j:
                # --- Diagonal: histogram ---
                fig.add_trace(
                    go.Histogram(
                        x=xi, nbinsx=30,
                        marker_color="#4C72B0",
                        opacity=0.75,
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
            elif i > j:
                # --- Lower triangle: 2-D density contour ---
                # Sub-sample if too many points to keep rendering fast
                _max_pts = 500
                if len(xi) > _max_pts:
                    idx = np.random.default_rng(42).choice(
                        len(xi), _max_pts, replace=False,
                    )
                    xi_s, yi_s = xi[idx], yi[idx]
                else:
                    xi_s, yi_s = xi, yi
                fig.add_trace(
                    go.Histogram2dContour(
                        x=xi_s, y=yi_s,
                        colorscale="Blues",
                        showscale=False,
                        ncontours=8,
                        contours_coloring="fill",
                        showlegend=False,
                        hovertemplate=(
                            f"{cols[j]}: %{{x:.2f}}<br>"
                            f"{cols[i]}: %{{y:.2f}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )
            else:
                # --- Upper triangle: correlation coefficient ---
                r_val = corr.iloc[i, j]
                fig.add_trace(
                    go.Scatter(
                        x=[0.5], y=[0.5],
                        mode="text",
                        text=[f"{r_val:.2f}"],
                        textfont=dict(
                            size=_corr_size(r_val),
                            color=_corr_color(r_val),
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"{cols[j]} vs {cols[i]}<br>"
                            f"r = {r_val:.3f}<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )
                fig.update_xaxes(
                    range=[0, 1], showticklabels=False,
                    showgrid=False, row=row, col=col,
                )
                fig.update_yaxes(
                    range=[0, 1], showticklabels=False,
                    showgrid=False, row=row, col=col,
                )

    # Axis labels: only left-most column (y) and bottom row (x)
    for i in range(n):
        # Left y-axis labels
        fig.update_yaxes(
            title_text=cols[i] if i != 0 or target is None else
            f"<b>{cols[i]}</b>",
            row=i + 1, col=1,
            title_font=dict(size=11),
            tickfont=dict(size=8),
        )
        # Bottom x-axis labels
        fig.update_xaxes(
            title_text=cols[i] if i != 0 or target is None else
            f"<b>{cols[i]}</b>",
            row=n, col=i + 1,
            title_font=dict(size=11),
            tickfont=dict(size=8),
        )
        # Hide tick labels on interior cells
        for j in range(n):
            if j > 0:
                fig.update_yaxes(
                    showticklabels=False, row=i + 1, col=j + 1,
                )
            if i < n - 1:
                fig.update_xaxes(
                    showticklabels=False, row=i + 1, col=j + 1,
                )

    cell_px = 150
    total = n * cell_px + 120
    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                "<span style='font-size:12px; color:#666;'>"
                "下三角: 2D密度等高線 | 対角: ヒストグラム | "
                "上三角: 相関係数 (赤=正, 青=負)"
                "</span>"
            ),
        ),
        template="plotly_white",
        height=max(700, total),
        width=max(700, total),
        showlegend=False,
    )
    return fig


def plotly_pairwise_scatter(
    features_df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    max_features: int = 6,
    max_samples: int = 300,
    title: str = "Pairwise Feature Plot (Top Variance)",
) -> go.Figure:
    """Lightweight pairwise feature plot using 2D density contours.

    Instead of ``px.scatter_matrix`` (which creates O(n_dim^2 * n_samples)
    WebGL points and can cause memory errors for large datasets), this
    implementation builds a custom subplot grid with:

    * **Off-diagonal**: ``go.Histogram2dContour`` — density contour, very
      lightweight even for thousands of samples.
    * **Diagonal**: ``go.Histogram`` — univariate distribution.

    A small random subsample of scatter points is overlaid so individual
    data points remain visible.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    target : pd.Series or None
        Optional target variable used for colour coding of scatter overlay.
    max_features : int
        Maximum number of features to include (picks highest-variance).
    max_samples : int
        Maximum number of scatter points to overlay (subsampled).
    title : str

    Returns
    -------
    go.Figure
    """
    from plotly.subplots import make_subplots

    if features_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[
            dict(text="No data", xref="paper", yref="paper",
                 x=0.5, y=0.5, showarrow=False, font_size=20)
        ])
        return fig

    # Select top-variance features for readability
    variances = features_df.var().sort_values(ascending=False)
    selected = list(variances.head(max_features).index)
    # Rebuild from C-contiguous numpy to avoid fragmented BlockManager (SIGSEGV risk)
    _sub_arr = np.ascontiguousarray(
        features_df[selected].to_numpy(dtype="float64", na_value=np.nan)
    )
    sub = pd.DataFrame(_sub_arr, columns=selected, index=features_df.index)
    n_dim = len(selected)

    # Short labels for display
    short = {c: c.replace("MagpieData ", "").replace("_", " ")
             for c in selected}

    # Subsample for scatter overlay to limit memory
    if len(sub) > max_samples:
        idx = sub.sample(max_samples, random_state=0).index
    else:
        idx = sub.index

    # Target colour array for subsample
    has_target = target is not None and len(target) == len(sub)
    if has_target:
        color_arr = np.ascontiguousarray(target.loc[idx].to_numpy(dtype="float64"))
        color_label = str(target.name) if target.name else "Target"
    else:
        color_arr = None
        color_label = None

    fig = make_subplots(
        rows=n_dim, cols=n_dim,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )

    for i in range(n_dim):
        for j in range(n_dim):
            row, col = i + 1, j + 1
            xi = selected[j]  # x-axis feature
            yi = selected[i]  # y-axis feature

            if i == j:
                # Diagonal — histogram
                fig.add_trace(
                    go.Histogram(
                        x=np.ascontiguousarray(sub[xi].to_numpy()),
                        nbinsx=30,
                        marker_color="rgba(76, 114, 176, 0.6)",
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
            else:
                # Off-diagonal — 2D density contour (lightweight)
                fig.add_trace(
                    go.Histogram2dContour(
                        x=np.ascontiguousarray(sub[xi].to_numpy()),
                        y=np.ascontiguousarray(sub[yi].to_numpy()),
                        colorscale="Blues",
                        showscale=False,
                        ncontours=10,
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
                # Scatter overlay (subsampled)
                scatter_kw: dict = dict(
                    x=np.ascontiguousarray(sub.loc[idx, xi].to_numpy()),
                    y=np.ascontiguousarray(sub.loc[idx, yi].to_numpy()),
                    mode="markers",
                    showlegend=False,
                )
                if color_arr is not None:
                    scatter_kw["marker"] = dict(
                        size=2, color=color_arr, colorscale="Viridis",
                        opacity=0.5,
                        showscale=(i == 0 and j == 1),
                        colorbar=dict(title=color_label)
                        if (i == 0 and j == 1) else None,
                    )
                else:
                    scatter_kw["marker"] = dict(
                        size=2, color="rgba(76, 114, 176, 0.4)",
                    )
                fig.add_trace(go.Scatter(**scatter_kw), row=row, col=col)

            # Axis labels on edges only
            if i == n_dim - 1:
                fig.update_xaxes(title_text=short[xi], row=row, col=col,
                                 title_font_size=9, tickfont_size=7)
            if j == 0:
                fig.update_yaxes(title_text=short[yi], row=row, col=col,
                                 title_font_size=9, tickfont_size=7)

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(500, n_dim * 110 + 80),
        width=max(600, n_dim * 120 + 80),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def build_summary_stats_md(
    compositions_df: Optional[pd.DataFrame],
    features_df: Optional[pd.DataFrame],
    target: Optional[pd.Series],
) -> str:
    """Build a Markdown summary of dataset statistics.

    Parameters
    ----------
    compositions_df : pd.DataFrame or None
    features_df : pd.DataFrame or None
    target : pd.Series or None

    Returns
    -------
    str
        Markdown text.
    """
    if compositions_df is None or target is None:
        return (
            "No dataset loaded yet.\n\n"
            "Generate data from the **Config** tab or upload a CSV "
            "using the **Upload** button above."
        )

    lines = ["### Dataset Overview\n"]
    lines.append(f"| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Samples | {len(target)} |")

    # Composition info
    non_zero = [
        c for c in compositions_df.columns
        if compositions_df[c].sum() > 0
    ]
    lines.append(f"| Elements | {len(non_zero)} ({', '.join(non_zero)}) |")

    # Feature info
    if features_df is not None:
        lines.append(f"| Features | {features_df.shape[1]} |")

    # Target stats
    lines.append(f"| Target | {target.name or 'unknown'} |")
    lines.append(f"| Target Min | {target.min():.2f} |")
    lines.append(f"| Target Max | {target.max():.2f} |")
    lines.append(f"| Target Mean | {target.mean():.2f} |")
    lines.append(f"| Target Std | {target.std():.2f} |")
    lines.append(f"| Target Median | {target.median():.2f} |")

    lines.append("")
    lines.append("### Feature Statistics (Top 10 by Variance)\n")
    if features_df is not None and not features_df.empty:
        variances = features_df.var().sort_values(ascending=False)
        top_features = list(variances.head(10).index)
        # Consolidate subset to single block before .describe()
        _sub = features_df[top_features]
        try:
            _arr = np.ascontiguousarray(
                _sub.to_numpy(dtype="float64", na_value=np.nan)
            )
            _sub = pd.DataFrame(_arr, columns=top_features, index=_sub.index)
        except (ValueError, TypeError):
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            desc = _sub.describe().round(3)
        try:
            lines.append(desc.to_markdown())
        except ImportError:
            # tabulate not installed — degrade gracefully
            lines.append(desc.to_string())
    else:
        lines.append("No features available.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 14. FS Comparison: Radar Chart (validity dimensions per feature set)
# ---------------------------------------------------------------------------

def plotly_fs_radar(
    scores: List[Any],
) -> go.Figure:
    """Radar (spider) chart comparing validity dimensions across feature sets.

    Each axis is one validity dimension; each feature set is a polygon.
    This makes it easy to see which FS is strong/weak on each axis.

    Parameters
    ----------
    scores : list of ValidityScore
        Sorted by total score (descending).

    Returns
    -------
    go.Figure
    """
    if not scores:
        fig = go.Figure()
        fig.update_layout(
            title="FS Radar — 実験を実行してください",
            annotations=[dict(
                text="No data", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font_size=20,
            )],
        )
        return fig

    dims = [
        ("effect_size", "Effect Size\n(効果量)"),
        ("stability", "Stability\n(安定性)"),
        ("generalisation", "Generalisation\n(汎化性)"),
        ("extrapolation_safety", "Extrap. Safety\n(外挿安全性)"),
    ]
    dim_keys = [d[0] for d in dims]
    dim_labels = [d[1] for d in dims]

    # Colour palette for up to 8 feature sets
    _colors = [
        "#4C72B0", "#55A868", "#C44E52", "#CCB974",
        "#8172B3", "#64B5CD", "#DD8452", "#A1C9F4",
    ]

    fig = go.Figure()
    for i, s in enumerate(scores):
        vals = [getattr(s, k) for k in dim_keys]
        # Close the polygon
        vals_closed = vals + [vals[0]]
        labels_closed = dim_labels + [dim_labels[0]]
        color = _colors[i % len(_colors)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)" if color.startswith("#") else None,
            opacity=0.7,
            name=f"{s.feature_set} (total={s.total:.3f})",
            line=dict(color=color, width=2),
            hovertemplate=(
                "%{theta}: %{r:.3f}<extra>" + s.feature_set + "</extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        title="特徴量セット比較レーダーチャート — FS Comparison Radar",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------------
# 15. FS Comparison: Box Plot (metric distribution per feature set)
# ---------------------------------------------------------------------------

def plotly_fs_boxplot(
    runs: List[Any],
    metric: str = "rmse_test",
) -> go.Figure:
    """Box plot showing metric distribution per feature set.

    Each box shows the spread of the metric across seeds/folds/splits
    for one feature set, making it easy to compare both central tendency
    and variability.

    Parameters
    ----------
    runs : list of RunResult
    metric : str
        Metric attribute name (default 'rmse_test').

    Returns
    -------
    go.Figure
    """
    _metric_labels = {
        "rmse_test": "RMSE (Test) — 小さいほど良い",
        "rmse_train": "RMSE (Train)",
        "mae_test": "MAE (Test) — 小さいほど良い",
        "mae_train": "MAE (Train)",
        "r2_test": "R$^2$ (Test) — 1に近いほど良い",
        "r2_train": "R$^2$ (Train)",
    }

    if not runs:
        fig = go.Figure()
        fig.update_layout(
            title="FS Box Plot — 実験を実行してください",
            annotations=[dict(
                text="No data", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font_size=20,
            )],
        )
        return fig

    records = []
    for r in runs:
        records.append({
            "feature_set": r.feature_set,
            "split_policy": r.split_policy,
            metric: getattr(r, metric, 0.0),
        })
    df = _records_to_df(records)

    _colors = [
        "#4C72B0", "#55A868", "#C44E52", "#CCB974",
        "#8172B3", "#64B5CD", "#DD8452", "#A1C9F4",
    ]
    fs_names = sorted(df["feature_set"].unique())

    fig = go.Figure()
    for i, fs in enumerate(fs_names):
        subset = df[df["feature_set"] == fs]
        color = _colors[i % len(_colors)]
        fig.add_trace(go.Box(
            y=subset[metric],
            name=fs,
            marker_color=color,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.5,
            hovertemplate=f"{fs}<br>{metric}: %{{y:.3f}}<extra></extra>",
        ))

    metric_display = _metric_labels.get(metric, metric.upper())
    fig.update_layout(
        title=f"特徴量セット別メトリクス分布 — {metric_display}",
        yaxis_title=metric_display,
        xaxis_title="特徴量セット (Feature Set)",
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 16. FS Comparison: Grouped Bar (mean metric per FS × split policy)
# ---------------------------------------------------------------------------

def plotly_fs_grouped_bar(
    runs: List[Any],
    metric: str = "rmse_test",
) -> go.Figure:
    """Grouped bar chart: mean metric per feature set, grouped by split policy.

    Parameters
    ----------
    runs : list of RunResult
    metric : str

    Returns
    -------
    go.Figure
    """
    _metric_labels = {
        "rmse_test": "RMSE (Test)",
        "rmse_train": "RMSE (Train)",
        "mae_test": "MAE (Test)",
        "mae_train": "MAE (Train)",
        "r2_test": "R$^2$ (Test)",
        "r2_train": "R$^2$ (Train)",
    }

    if not runs:
        fig = go.Figure()
        fig.update_layout(title="FS Grouped Bar — 実験を実行してください")
        return fig

    records = []
    for r in runs:
        records.append({
            "feature_set": r.feature_set,
            "split_policy": r.split_policy,
            metric: getattr(r, metric, 0.0),
        })
    df = _records_to_df(records)
    pivot = df.groupby(["feature_set", "split_policy"])[metric].mean().unstack(fill_value=0)

    _split_colors = {
        "RandomCV": "#4C72B0",
        "CompositionBlock": "#55A868",
        "ElementExclusion": "#C44E52",
    }

    fig = go.Figure()
    for sp_col in pivot.columns:
        color = _split_colors.get(sp_col, "#8C8C8C")
        fig.add_trace(go.Bar(
            x=list(pivot.index),
            y=np.ascontiguousarray(pivot[sp_col].to_numpy(dtype="float64")),
            name=sp_col,
            marker_color=color,
            hovertemplate=f"{sp_col}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))

    metric_display = _metric_labels.get(metric, metric.upper())
    fig.update_layout(
        barmode="group",
        title=f"特徴量セット × 分割方法 — {metric_display}",
        xaxis_title="特徴量セット (Feature Set)",
        yaxis_title=metric_display,
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# 17. FS Comparison: Summary Table (Markdown)
# ---------------------------------------------------------------------------

def build_fs_comparison_summary_md(
    runs: List[Any],
    scores: List[Any],
) -> str:
    """Build a Markdown summary comparing feature sets.

    Includes: mean/std of RMSE(Test), R$^2$(Test) per FS,
    validity ranking, and a recommendation.

    Parameters
    ----------
    runs : list of RunResult
    scores : list of ValidityScore

    Returns
    -------
    str  Markdown text.
    """
    if not runs or not scores:
        return (
            "### FS Comparison Summary\n\n"
            "実験を実行すると、ここに特徴量セットの比較結果が表示されます。"
        )

    # Group runs by feature set
    fs_data: Dict[str, List[Any]] = {}
    for r in runs:
        fs_data.setdefault(r.feature_set, []).append(r)

    lines = [
        "### FS Comparison Summary — 特徴量セット比較サマリー\n",
        "| FS | 特徴量数 | RMSE(Test) Mean | RMSE(Test) Std | "
        "R$^2$(Test) Mean | Validity Total | 推奨 |",
        "|---|---|---|---|---|---|---|",
    ]

    # Feature set sizes (approximate)
    _fs_sizes = {
        "FS_BASE": 8, "FS_THERMO": 11, "FS_SIZE": 12,
        "FS_ELECTRON": 11, "FS_ALL": 18, "FS_MAGPIE": 132,
    }

    # Build score lookup
    score_map = {s.feature_set: s for s in scores}
    best_fs = scores[0].feature_set if scores else ""

    for fs_name in sorted(fs_data.keys()):
        fs_runs = fs_data[fs_name]
        rmses = [r.rmse_test for r in fs_runs if r.rmse_test > 0]
        r2s = [r.r2_test for r in fs_runs]
        n_feat = _fs_sizes.get(fs_name, "?")
        rmse_mean = f"{np.mean(rmses):.2f}" if rmses else "N/A"
        rmse_std = f"{np.std(rmses):.2f}" if len(rmses) > 1 else "N/A"
        r2_mean = f"{np.mean(r2s):.4f}" if r2s else "N/A"
        vs = score_map.get(fs_name)
        total = f"{vs.total:.4f}" if vs else "N/A"
        recommend = "**Best**" if fs_name == best_fs else ""
        lines.append(
            f"| {fs_name} | {n_feat} | {rmse_mean} | {rmse_std} | "
            f"{r2_mean} | {total} | {recommend} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Recommendation text
    if scores:
        best = scores[0]
        lines.append(f"**推奨特徴量セット: {best.feature_set}**\n")
        lines.append(
            f"- 総合妥当性スコア: {best.total:.4f}\n"
            f"- 効果量 (Effect Size): {best.effect_size:.4f}\n"
            f"- 安定性 (Stability): {best.stability:.4f}\n"
            f"- 汎化性 (Generalisation): {best.generalisation:.4f}\n"
            f"- 外挿安全性 (Extrap. Safety): {best.extrapolation_safety:.4f}\n"
            f"- リークペナルティ: {best.leak_penalty:.4f}"
        )
        lines.append("")

        # Interpretation guide
        lines.append("#### 判断のポイント\n")
        if best.total >= 0.6:
            lines.append(
                "- Total Score ≥ 0.6: **良好**。この特徴量セットは安定した予測性能を持ち、"
                "外挿にも比較的安全です。"
            )
        elif best.total >= 0.4:
            lines.append(
                "- Total Score 0.4〜0.6: **中程度**。改善の余地があります。"
                "他の特徴量セットとの組み合わせを検討してください。"
            )
        else:
            lines.append(
                "- Total Score < 0.4: **要改善**。特徴量の選択を見直すか、"
                "データの質を確認してください。"
            )

        if best.leak_penalty > 0.1:
            lines.append(
                f"\n- **注意**: Leak Penalty = {best.leak_penalty:.4f} > 0.1。"
                "データリークの可能性があります。分割方法を変えて再評価してください。"
            )

        # Compare best vs MAGPIE
        magpie_score = score_map.get("FS_MAGPIE")
        if magpie_score and best.feature_set != "FS_MAGPIE":
            if magpie_score.total > best.total - 0.05:
                lines.append(
                    f"\n- **参考**: FS\\_MAGPIE (total={magpie_score.total:.4f}) も"
                    "同程度のスコアです。132個の特徴量を使う大規模セットで、"
                    "データ量が十分であれば高い予測性能が期待できます。"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 18. FS Comparison: Train vs Test metric scatter
# ---------------------------------------------------------------------------

def plotly_fs_train_vs_test(
    runs: List[Any],
    metric_base: str = "rmse",
) -> go.Figure:
    """Scatter plot of train metric vs test metric per feature set.

    Points near the diagonal = no overfitting.
    Points far above = overfitting (train << test).

    Parameters
    ----------
    runs : list of RunResult
    metric_base : str
        Base metric name ('rmse', 'mae', 'r2'). Train and test
        columns are inferred as ``{base}_train`` / ``{base}_test``.

    Returns
    -------
    go.Figure
    """
    train_key = f"{metric_base}_train"
    test_key = f"{metric_base}_test"

    _metric_labels = {
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R$^2$",
    }
    label = _metric_labels.get(metric_base, metric_base.upper())

    if not runs:
        fig = go.Figure()
        fig.update_layout(title=f"Train vs Test {label} — 実験を実行してください")
        return fig

    _colors = {
        "FS_BASE": "#4C72B0", "FS_THERMO": "#55A868",
        "FS_SIZE": "#C44E52", "FS_ELECTRON": "#CCB974",
        "FS_ALL": "#8172B3", "FS_MAGPIE": "#64B5CD",
    }
    _symbols = {
        "RandomCV": "circle", "CompositionBlock": "square",
        "ElementExclusion": "diamond",
    }

    fig = go.Figure()

    # Group by feature set
    fs_groups: Dict[str, List[Any]] = {}
    for r in runs:
        fs_groups.setdefault(r.feature_set, []).append(r)

    for fs_name, fs_runs in sorted(fs_groups.items()):
        trains = [getattr(r, train_key, 0.0) for r in fs_runs]
        tests = [getattr(r, test_key, 0.0) for r in fs_runs]
        splits = [r.split_policy for r in fs_runs]
        symbols = [_symbols.get(sp, "circle") for sp in splits]
        color = _colors.get(fs_name, "#8C8C8C")

        fig.add_trace(go.Scatter(
            x=trains,
            y=tests,
            mode="markers",
            marker=dict(color=color, size=8, opacity=0.7),
            name=fs_name,
            customdata=list(zip(splits, [r.seed for r in fs_runs])),
            hovertemplate=(
                f"{fs_name}<br>"
                f"{label}(Train): %{{x:.3f}}<br>"
                f"{label}(Test): %{{y:.3f}}<br>"
                "Split: %{customdata[0]}<br>"
                "Seed: %{customdata[1]}<extra></extra>"
            ),
        ))

    # Diagonal (y=x)
    all_vals = []
    for r in runs:
        all_vals.append(getattr(r, train_key, 0.0))
        all_vals.append(getattr(r, test_key, 0.0))
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        margin = (hi - lo) * 0.05 if hi > lo else 0.1
        fig.add_trace(go.Scatter(
            x=[lo - margin, hi + margin],
            y=[lo - margin, hi + margin],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="y = x (過学習なし)",
            showlegend=True,
        ))

    fig.update_layout(
        title=f"Train vs Test {label} — 対角線から離れるほど過学習の兆候",
        xaxis_title=f"{label} (Train)",
        yaxis_title=f"{label} (Test)",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------------
# 19. Knowledge Graph Visualization
# ---------------------------------------------------------------------------

def plotly_knowledge_graph(
    kg: Any,
    layout: str = "spring",
    highlight_type: Optional[str] = None,
) -> go.Figure:
    """Interactive Plotly visualization of the knowledge graph.

    Uses NetworkX layout algorithms to position nodes, then renders
    with Plotly scatter + lines.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph instance.
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'shell'.
    highlight_type : str, optional
        If set, highlight nodes of this type.

    Returns
    -------
    go.Figure
    """
    try:
        import networkx as nx
    except ImportError:
        fig = go.Figure()
        fig.update_layout(
            title="Knowledge Graph — networkxが必要です",
            annotations=[dict(
                text="pip install networkx", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font_size=16,
            )],
        )
        return fig

    graph = kg.graph
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Knowledge Graph — ノードなし",
            annotations=[dict(
                text="実験を実行するとグラフが構築されます",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font_size=16,
            )],
        )
        return fig

    # Compute layout positions
    _layout_funcs = {
        "spring": lambda g: nx.spring_layout(g, k=2.0, iterations=50, seed=42),
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout,
    }
    layout_fn = _layout_funcs.get(layout, _layout_funcs["spring"])
    try:
        pos = layout_fn(graph)
    except Exception:
        pos = nx.spring_layout(graph, k=2.0, iterations=50, seed=42)

    # Node type -> color and size
    _type_colors = {
        "experiment": "#4C72B0",
        "feature_set": "#55A868",
        "feature": "#CCB974",
        "model": "#C44E52",
        "paper": "#8172B3",
        "workflow": "#64B5CD",
        "split_policy": "#DD8452",
    }
    _type_sizes = {
        "experiment": 8,
        "feature_set": 18,
        "feature": 10,
        "model": 16,
        "paper": 14,
        "workflow": 12,
        "split_policy": 14,
    }

    fig = go.Figure()

    # Draw edges first (as lines)
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Draw nodes grouped by type (for legend)
    type_groups: Dict[str, list] = {}
    for node_id in graph.nodes():
        ntype = graph.nodes[node_id].get("node_type", "unknown")
        type_groups.setdefault(ntype, []).append(node_id)

    _type_labels_jp = {
        "experiment": "実験ラン",
        "feature_set": "特徴量セット",
        "feature": "個別特徴量",
        "model": "モデル",
        "paper": "論文",
        "workflow": "文献WF",
        "split_policy": "分割方法",
    }

    for ntype, node_ids in type_groups.items():
        xs = [pos[nid][0] for nid in node_ids if nid in pos]
        ys = [pos[nid][1] for nid in node_ids if nid in pos]
        labels = [
            graph.nodes[nid].get("label", nid) for nid in node_ids if nid in pos
        ]
        color = _type_colors.get(ntype, "#8C8C8C")
        size = _type_sizes.get(ntype, 10)
        display_name = _type_labels_jp.get(ntype, ntype)

        # Highlight effect
        if highlight_type and ntype == highlight_type:
            size = size + 6
            opacity = 1.0
        elif highlight_type:
            opacity = 0.3
        else:
            opacity = 0.8

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if size >= 14 else "markers",
            marker=dict(
                color=color, size=size, opacity=opacity,
                line=dict(width=1, color="white"),
            ),
            text=labels if size >= 14 else None,
            textposition="top center",
            textfont=dict(size=9),
            name=f"{display_name} ({len(node_ids)})",
            hovertext=labels,
            hovertemplate="%{hovertext}<extra>" + display_name + "</extra>",
        ))

    fig.update_layout(
        title="知識グラフ — Knowledge Graph",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=650,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
    )
    return fig
