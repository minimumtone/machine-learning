"""
Feature Selection Module for Extrapolation Discovery Platform
特徴量選択モジュール

Implements four model selection / feature selection algorithms:
  - Lasso (L1 regularisation): sklearn LassoCV
  - AIC  (Akaike Information Criterion): forward stepwise selection
  - BIC  (Bayesian Information Criterion): forward stepwise selection
  - ARD  (Automatic Relevance Determination): Bayesian sparse regression

Each method returns a ranked list of selected features with importance
scores.  The ``run_feature_selection`` convenience function runs all
(or a subset of) methods and returns a consolidated result.

Physical interpretation notes are embedded in ``FS_PHYSICAL_ORIGINS``
for downstream display in the GUI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression, LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical origin descriptions for each feature set
# ---------------------------------------------------------------------------

FS_PHYSICAL_ORIGINS: Dict[str, str] = {
    "FS_BASE": (
        "基本的な組成記述子（原子半径差 delta_r、混合エントロピー S_mix、"
        "VEC、電気陰性度差 delta_en など）。"
        "固溶体の格子歪みやHume-Rothery則に直接対応し、"
        "HEAの単相安定性と強化機構の第一近似を捉える。"
    ),
    "FS_THERMO": (
        "熱力学的安定性指標（混合エンタルピー H_mix、Omega パラメータ、"
        "固溶体指数 ss_index、相分離リスク phase_sep_risk など）。"
        "Gibbsの自由エネルギー競合に基づき、固溶体形成 vs 金属間化合物析出の"
        "傾向を記述する。Omega > 1.1 かつ delta_r < 6.6% で"
        "単相固溶体が形成されやすい（Zhang et al., 2012）。"
    ),
    "FS_SIZE": (
        "原子サイズ・弾性ミスマッチ特徴量（delta_r_percent、"
        "modulus_mismatch、volume_mismatch など）。"
        "固溶体強化（SSH: Solid-Solution Hardening）のLabusch-Varvenne"
        "モデルに対応し、格子歪みエネルギーと転位-溶質相互作用を記述する。"
        "サイズ差が大きいほど強化量が増大するが、過大な差は相分離を誘発する。"
    ),
    "FS_ELECTRON": (
        "電子構造プロキシ特徴量（d電子数、DOS近似、VEC分散、"
        "d_electron_concentration など）。"
        "d-d相互作用による結合強化とPeierls障壁の変化を反映する。"
        "BCC HEAではVEC < 6.87で安定（Guo et al., 2011）、"
        "FCC HEAではVEC > 8.0で安定となる電子濃度規則に対応。"
    ),
    "FS_ALL": (
        "全ドメイン特徴量の和集合（BASE + THERMO + SIZE + ELECTRON）。"
        "多重共線性により個別特徴量の寄与が分散する可能性があるが、"
        "非線形モデル（XGBoost等）では特徴量間の交互作用を捉えることで"
        "予測精度が向上する場合がある。"
    ),
    "FS_MAGPIE": (
        "MAGPIE記述子（Ward et al., npj Comput. Mater. 2016）: "
        "22元素物性 x 6統計量 = 132次元。"
        "matminer ElementPropertyフィーチャライザ互換。"
        "網羅的な元素物性統計により、未知の物性相関を捉える探索的アプローチ。"
        "次元の呪い（curse of dimensionality）に注意が必要だが、"
        "適切な正則化（Lasso/ARD）で有効特徴量を抽出できる。"
    ),
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionResult:
    """Result of a single feature selection method.

    Attributes
    ----------
    method : str
        Method name (e.g. "Lasso", "AIC", "BIC", "ARD").
    selected_features : list of str
        Feature names selected by this method.
    importance_scores : dict of {feature_name: float}
        Importance / coefficient magnitude for each selected feature.
    all_scores : dict of {feature_name: float}
        Scores for *all* input features (not just selected ones).
    n_selected : int
        Number of selected features.
    """
    method: str
    selected_features: List[str] = field(default_factory=list)
    importance_scores: Dict[str, float] = field(default_factory=dict)
    all_scores: Dict[str, float] = field(default_factory=dict)
    n_selected: int = 0


@dataclass
class FeatureSelectionSummary:
    """Consolidated feature selection results across multiple methods.

    Attributes
    ----------
    results : dict of {method_name: FeatureSelectionResult}
    consensus_features : list of str
        Features selected by >= ``consensus_threshold`` methods.
    consensus_threshold : int
        Minimum number of methods that must select a feature for consensus.
    feature_set : str
        Name of the feature set that was analysed.
    """
    results: Dict[str, FeatureSelectionResult] = field(default_factory=dict)
    consensus_features: List[str] = field(default_factory=list)
    consensus_threshold: int = 2
    feature_set: str = ""


# ---------------------------------------------------------------------------
# Individual feature selection methods
# ---------------------------------------------------------------------------

def _run_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> FeatureSelectionResult:
    """L1-regularised regression (LassoCV) for feature selection."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = LassoCV(cv=min(cv, len(X)), max_iter=10000, random_state=42)
    model.fit(X_scaled, y.values)

    coefs = np.abs(model.coef_)
    all_scores = dict(zip(X.columns, coefs.tolist()))

    # Selected = non-zero coefficients
    selected_mask = coefs > 1e-8
    selected = [col for col, sel in zip(X.columns, selected_mask) if sel]
    importance = {col: float(coefs[i]) for i, col in enumerate(X.columns) if selected_mask[i]}

    return FeatureSelectionResult(
        method="Lasso",
        selected_features=selected,
        importance_scores=importance,
        all_scores=all_scores,
        n_selected=len(selected),
    )


def _run_aic_forward(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 20,
) -> FeatureSelectionResult:
    """Forward stepwise selection using AIC (Akaike Information Criterion).

    AIC = n * ln(RSS/n) + 2k
    where n = sample size, k = number of parameters.
    Lower AIC is better. AIC penalises complexity less than BIC,
    so it tends to select more features.
    """
    return _forward_stepwise_ic(X, y, criterion="aic", max_features=max_features)


def _run_bic_forward(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 20,
) -> FeatureSelectionResult:
    """Forward stepwise selection using BIC (Bayesian Information Criterion).

    BIC = n * ln(RSS/n) + k * ln(n)
    BIC penalises model complexity more strongly than AIC,
    so it tends to select fewer features — preferring parsimony.
    """
    return _forward_stepwise_ic(X, y, criterion="bic", max_features=max_features)


def _forward_stepwise_ic(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str = "aic",
    max_features: int = 20,
) -> FeatureSelectionResult:
    """Forward stepwise feature selection with information criterion."""
    from sklearn.linear_model import LinearRegression

    n = len(y)
    remaining = list(X.columns)
    selected: List[str] = []
    best_ic = np.inf
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.values), columns=X.columns, index=X.index,
    )
    y_arr = y.values

    for step in range(min(max_features, len(remaining))):
        best_feat = None
        best_ic_step = np.inf

        for feat in remaining:
            trial = selected + [feat]
            X_trial = X_scaled[trial].values
            model = LinearRegression()
            model.fit(X_trial, y_arr)
            y_pred = model.predict(X_trial)
            rss = float(np.sum((y_arr - y_pred) ** 2))
            k = len(trial) + 1  # +1 for intercept

            if rss <= 0:
                ic = -np.inf
            elif criterion == "aic":
                ic = n * np.log(rss / n) + 2 * k
            else:  # bic
                ic = n * np.log(rss / n) + k * np.log(n)

            if ic < best_ic_step:
                best_ic_step = ic
                best_feat = feat

        if best_feat is None or best_ic_step >= best_ic:
            break  # no improvement

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_ic = best_ic_step

    # Compute importance as order of selection (first selected = most important)
    importance = {}
    for rank, feat in enumerate(selected):
        importance[feat] = float(len(selected) - rank) / len(selected) if selected else 0.0

    all_scores: Dict[str, float] = {col: 0.0 for col in X.columns}
    all_scores.update(importance)

    method_name = "AIC" if criterion == "aic" else "BIC"
    return FeatureSelectionResult(
        method=method_name,
        selected_features=selected,
        importance_scores=importance,
        all_scores=all_scores,
        n_selected=len(selected),
    )


def _run_ard(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.01,
) -> FeatureSelectionResult:
    """Automatic Relevance Determination (Bayesian sparse regression).

    ARD learns per-feature precision (inverse variance) hyperparameters.
    Features whose precision grows very large are effectively pruned.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = ARDRegression(max_iter=500, compute_score=True)
    model.fit(X_scaled, y.values)

    coefs = np.abs(model.coef_)
    all_scores = dict(zip(X.columns, coefs.tolist()))

    # Select features with abs(coef) above threshold
    max_coef = coefs.max() if coefs.max() > 0 else 1.0
    normalised = coefs / max_coef
    selected_mask = normalised > threshold
    selected = [col for col, sel in zip(X.columns, selected_mask) if sel]
    importance = {col: float(coefs[i]) for i, col in enumerate(X.columns) if selected_mask[i]}

    return FeatureSelectionResult(
        method="ARD",
        selected_features=selected,
        importance_scores=importance,
        all_scores=all_scores,
        n_selected=len(selected),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METHOD_MAP = {
    "Lasso": _run_lasso,
    "AIC": _run_aic_forward,
    "BIC": _run_bic_forward,
    "ARD": _run_ard,
}

AVAILABLE_METHODS: List[str] = list(_METHOD_MAP.keys())


def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    methods: Optional[List[str]] = None,
    consensus_threshold: int = 2,
    feature_set: str = "",
) -> FeatureSelectionSummary:
    """Run feature selection with one or more methods.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    methods : list of str, optional
        Methods to run. Defaults to all: ["Lasso", "AIC", "BIC", "ARD"].
    consensus_threshold : int
        Minimum methods that must select a feature for it to be consensus.
    feature_set : str
        Name of the feature set being analysed (for display purposes).

    Returns
    -------
    FeatureSelectionSummary
    """
    if methods is None:
        methods = AVAILABLE_METHODS

    results: Dict[str, FeatureSelectionResult] = {}
    for method_name in methods:
        func = _METHOD_MAP.get(method_name)
        if func is None:
            logger.warning("Unknown feature selection method: %s", method_name)
            continue
        try:
            result = func(X, y)
            results[method_name] = result
            logger.info(
                "Feature selection [%s] on %s: %d / %d features selected",
                method_name, feature_set, result.n_selected, X.shape[1],
            )
        except Exception:
            logger.exception("Feature selection [%s] failed on %s", method_name, feature_set)

    # Compute consensus
    from collections import Counter
    feature_counts: Counter = Counter()
    for res in results.values():
        for feat in res.selected_features:
            feature_counts[feat] += 1

    consensus = [
        feat for feat, count in feature_counts.most_common()
        if count >= consensus_threshold
    ]

    return FeatureSelectionSummary(
        results=results,
        consensus_features=consensus,
        consensus_threshold=consensus_threshold,
        feature_set=feature_set,
    )
