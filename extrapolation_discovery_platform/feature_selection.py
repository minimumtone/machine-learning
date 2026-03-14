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
    # CRITICAL: Force C-contiguous layout before BLAS calls.
    # pandas 3.0 .values on fragmented DataFrames returns F-contiguous
    # arrays which cause SIGSEGV in BLAS/LAPACK routines.
    X_scaled = scaler.fit_transform(
        np.ascontiguousarray(X.to_numpy(dtype="float64", na_value=np.nan))
    )

    model = LassoCV(cv=min(cv, len(X)), max_iter=10000, random_state=42)
    model.fit(X_scaled, np.ascontiguousarray(y.to_numpy(dtype="float64")))

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
    """Forward stepwise selection using CV-based error with AIC-style stopping.

    Uses cross-validated MSE as the selection criterion.  CV already
    penalises overfitting via out-of-sample evaluation, so no explicit
    AIC penalty term is added (that would double-penalise complexity).
    The 'aic' label is retained for backward compatibility; the method
    is less conservative than BIC-style (more CV folds → larger training
    sets per fold → easier to detect marginal improvements → more features).
    """
    return _forward_stepwise_ic(X, y, criterion="aic", max_features=max_features)


def _run_bic_forward(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 20,
) -> FeatureSelectionResult:
    """Forward stepwise selection using CV-based error with BIC-style stopping.

    Uses cross-validated MSE as the selection criterion.  Compared to the
    AIC variant this uses fewer CV folds, meaning smaller training sets
    per fold and noisier error estimates — making marginal feature gains
    harder to detect, so the procedure stops earlier with fewer features,
    preferring parsimony.  The 'bic' label is retained for backward
    compatibility.
    """
    return _forward_stepwise_ic(X, y, criterion="bic", max_features=max_features)


_AIC_BIC_MAX_FEATURES = 30  # Skip AIC/BIC forward stepwise when n_features > this


def _forward_stepwise_ic(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str = "aic",
    max_features: int = 20,
) -> Optional[FeatureSelectionResult]:
    """Forward stepwise feature selection with information criterion.

    Returns None if n_features > _AIC_BIC_MAX_FEATURES to avoid O(n²)
    bottleneck on high-dimensional feature sets (e.g. FS_MAGPIE with 132+
    features).
    """
    if X.shape[1] > _AIC_BIC_MAX_FEATURES:
        method_name = "AIC" if criterion == "aic" else "BIC"
        logger.info(
            "%s skipped: %d features > threshold %d (O(n²) bottleneck)",
            method_name, X.shape[1], _AIC_BIC_MAX_FEATURES,
        )
        return None

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    n = len(y)
    remaining = list(X.columns)
    selected: List[str] = []
    best_score = np.inf
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(
            np.ascontiguousarray(X.to_numpy(dtype="float64", na_value=np.nan))
        ),
        columns=X.columns, index=X.index,
    )
    y_arr = np.ascontiguousarray(y.to_numpy(dtype="float64"))

    # Use cross-validated MSE directly as the selection score.
    # CV already penalises overfitting through out-of-sample evaluation,
    # so we do NOT layer an explicit AIC/BIC penalty on top (that would
    # double-penalise complexity).  We differentiate "aic" vs "bic" by
    # fold count: more folds → larger training sets → easier to detect
    # marginal improvements → more features selected (AIC = permissive).
    # Fewer folds → smaller training sets → noisier estimates → stops
    # earlier with fewer features (BIC = conservative / parsimonious).
    if criterion == "aic":
        cv_folds = min(n, 10)      # 10-fold (more data per fold → more features)
    else:  # bic — fewer folds → more conservative
        cv_folds = min(n, 5)       # 5-fold (less data per fold → fewer features)

    for step in range(min(max_features, len(remaining))):
        best_feat = None
        best_score_step = np.inf

        for feat in remaining:
            trial = selected + [feat]
            X_trial = np.ascontiguousarray(
                X_scaled[trial].to_numpy(dtype="float64", na_value=np.nan)
            )
            model = LinearRegression()
            # CV-based MSE (negative by sklearn convention)
            cv_mse = -cross_val_score(
                model, X_trial, y_arr,
                cv=cv_folds, scoring="neg_mean_squared_error",
            ).mean()

            if cv_mse < best_score_step:
                best_score_step = cv_mse
                best_feat = feat

        if best_feat is None or best_score_step >= best_score:
            break  # no improvement

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_score_step

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
    X_scaled = scaler.fit_transform(
        np.ascontiguousarray(X.to_numpy(dtype="float64", na_value=np.nan))
    )

    model = ARDRegression(max_iter=500, compute_score=True)
    model.fit(X_scaled, np.ascontiguousarray(y.to_numpy(dtype="float64")))

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
            if result is None:
                # Method chose to skip (e.g. AIC/BIC with too many features)
                continue
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
