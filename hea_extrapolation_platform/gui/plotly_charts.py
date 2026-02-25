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
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

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
    pca = PCA(n_components=2)
    X_all = pd.concat([X_train, X_query], axis=0, ignore_index=True)
    coords = pca.fit_transform(X_all.values)
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
        xaxis_title=f"PC1 ({var_ratio[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({var_ratio[1]*100:.1f}%)",
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
        title="Feature Set Validity Ranking",
        xaxis_title="Score",
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
    df = pd.DataFrame(records)
    pivot = df.groupby(["feature_set", "split_policy"])[metric].mean().unstack(fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="YlOrRd",
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        hovertemplate=(
            "Feature Set: %{y}<br>Split: %{x}<br>"
            f"{metric}: %{{z:.2f}}<extra></extra>"
        ),
        colorbar=dict(title=metric.upper()),
    ))

    fig.update_layout(
        title=f"Performance Comparison ({metric})",
        xaxis_title="Split Policy",
        yaxis_title="Feature Set",
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
        xaxis_title="True Value",
        yaxis_title="Predicted Value",
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
        title="Uncertainty vs OOD Score",
        xaxis_title="OOD Score",
        yaxis_title="Prediction Uncertainty (std)",
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
        xaxis_title="Count (papers)",
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
    return pd.DataFrame(records)


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
    return pd.DataFrame(records)


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
        x=target.values,
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
        y=list(means.values),
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
    max_features: int = 15,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Correlation heatmap of selected features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    max_features : int
        Maximum number of features to show (picks highest-variance).
    title : str

    Returns
    -------
    go.Figure
    """
    if features_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Select top-variance features for readability
    variances = features_df.var().sort_values(ascending=False)
    selected = list(variances.head(max_features).index)
    sub = features_df[selected]
    corr = sub.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate=(
            "%{y} vs %{x}<br>Corr: %{z:.3f}<extra></extra>"
        ),
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(450, len(selected) * 35 + 100),
        width=max(500, len(selected) * 40 + 100),
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
        desc = features_df[top_features].describe().round(3)
        lines.append(desc.to_markdown())
    else:
        lines.append("No features available.")

    return "\n".join(lines)
