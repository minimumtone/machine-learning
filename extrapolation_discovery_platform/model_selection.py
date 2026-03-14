"""
Nested Cross-Validation & Automated Model Selection
ネステッドCV + 自動モデル選択モジュール

Implements:
  A. Target binning for StratifiedKFold in regression tasks
  B. Nested CV (outer=evaluation, inner=HPO)
  C. Feature selection as pipeline step (SelectFromModel + LassoCV/ARD/Ridge)
  D. Candidate pipeline definitions (linear + tree models)
  E. Model comparison with sparsity preference
  F. Final model refit & save (joblib + metadata JSON)

Design:
  - Outer CV uses StratifiedKFold on binned target for balanced evaluation
  - Inner CV uses RandomizedSearchCV for hyperparameter optimisation
  - SelectFromModel wraps LassoCV/ARD/RidgeCV for feature selection *inside*
    the pipeline, preventing data leakage
  - Best model is chosen by outer-fold mean RMSE; ties broken by sparsity
  - Final model is retrained on full data and saved with metadata
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ARDRegression, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from scipy.stats import loguniform, randint, uniform

from extrapolation_discovery_platform._utils import get_safe_n_jobs, safe_array

logger = logging.getLogger(__name__)


# Try importing optional packages
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False


# ---------------------------------------------------------------------------
# A. Target binning for StratifiedKFold
# ---------------------------------------------------------------------------

def make_stratify_labels(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretise continuous target into quantile bins for StratifiedKFold.

    Parameters
    ----------
    y : 1-D array-like
        Target values.
    n_bins : int
        Number of bins (default 5).  Automatically reduced when there are
        fewer unique values.

    Returns
    -------
    labels : ndarray of int, shape (n_samples,)
        Bin labels 0 .. n_bins-1.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n_unique = len(np.unique(y_arr))
    bins = min(n_bins, max(2, n_unique))

    kb = KBinsDiscretizer(
        n_bins=bins, encode="ordinal", strategy="quantile",
        subsample=None,
    )
    labels = kb.fit_transform(y_arr).astype(int).ravel()
    return labels


# ---------------------------------------------------------------------------
# B. Result containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Per-candidate result from nested CV."""
    name: str
    outer_scores: List[float] = field(default_factory=list)
    best_params_per_fold: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    models: List[Any] = field(default_factory=list)
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    median_n_features: Optional[int] = None


@dataclass
class ModelSelectionResult:
    """Summary of the full model selection process."""
    best_name: str = ""
    best_mean_rmse: float = float("inf")
    best_std_rmse: float = 0.0
    best_params: Optional[Dict[str, Any]] = None
    selected_features: Optional[List[str]] = None
    n_features_selected: Optional[int] = None
    all_candidates: List[CandidateResult] = field(default_factory=list)
    model_path: Optional[str] = None
    meta_path: Optional[str] = None
    elapsed_sec: float = 0.0

    def summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "best_model": self.best_name,
            "mean_rmse": round(self.best_mean_rmse, 6),
            "std_rmse": round(self.best_std_rmse, 6),
            "n_features_selected": self.n_features_selected,
            "selected_features": self.selected_features,
            "best_params": self.best_params,
            "all_candidates": [
                {
                    "name": c.name,
                    "mean_rmse": round(c.mean_rmse, 6),
                    "std_rmse": round(c.std_rmse, 6),
                    "median_n_features": c.median_n_features,
                }
                for c in self.all_candidates
            ],
            "model_path": self.model_path,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }

    def summary_markdown(self) -> str:
        """Return a Markdown summary for GUI display."""
        lines = [
            "## Model Selection Results (Nested CV)",
            "",
            f"**Best model**: `{self.best_name}`",
            f"- Outer CV RMSE: {self.best_mean_rmse:.4f} ± {self.best_std_rmse:.4f}",
        ]
        if self.n_features_selected is not None:
            lines.append(f"- Selected features: {self.n_features_selected}")
        if self.best_params:
            lines.append(f"- Best hyperparameters: `{self.best_params}`")
        lines.append(f"- Elapsed: {self.elapsed_sec:.1f} sec")
        lines.append("")

        # Comparison table
        lines.append("### Candidate Comparison")
        lines.append("")
        lines.append("| Model | Mean RMSE | Std RMSE | Features | Rank |")
        lines.append("|---|---|---|---|---|")
        sorted_candidates = sorted(
            self.all_candidates, key=lambda c: c.mean_rmse,
        )
        for rank, c in enumerate(sorted_candidates, 1):
            n_feat = str(c.median_n_features) if c.median_n_features is not None else "all"
            marker = " **Best**" if c.name == self.best_name else ""
            lines.append(
                f"| {c.name}{marker} | {c.mean_rmse:.4f} "
                f"| {c.std_rmse:.4f} | {n_feat} | {rank} |"
            )
        lines.append("")

        if self.selected_features:
            lines.append("### Selected Features")
            lines.append("")
            for i, feat in enumerate(self.selected_features, 1):
                lines.append(f"{i}. `{feat}`")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# C. Candidate pipeline definitions
# ---------------------------------------------------------------------------

def build_candidate_pipelines(
    n_features: int,
    quick: bool = False,
) -> List[Tuple[str, Pipeline, Optional[Dict[str, Any]]]]:
    """Build candidate (name, pipeline, param_dist) tuples.

    Parameters
    ----------
    n_features : int
        Number of input features (used to size param grids).
    quick : bool
        If True, use reduced search spaces.

    Returns
    -------
    candidates : list of (name, Pipeline, param_dist_or_None)
    """
    candidates: List[Tuple[str, Pipeline, Optional[Dict[str, Any]]]] = []

    # 1. Lasso (sparse linear — LassoCV selects alpha internally)
    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("est", LassoCV(
            alphas=np.logspace(-6, 2, 30 if quick else 50),
            cv=3, max_iter=10000, n_jobs=1,
        )),
    ])
    candidates.append(("Lasso", pipe_lasso, None))

    # 2. Ridge (L2 linear — RidgeCV selects alpha internally)
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("est", RidgeCV(
            alphas=np.logspace(-3, 6, 20),
            cv=None,  # efficient LOO
        )),
    ])
    candidates.append(("Ridge", pipe_ridge, None))

    # 3. ARD (Bayesian sparse)
    pipe_ard = Pipeline([
        ("scaler", StandardScaler()),
        ("est", ARDRegression(max_iter=300)),
    ])
    candidates.append(("ARD", pipe_ard, None))

    # 4. Lasso-selector → XGBoost
    if _XGB_AVAILABLE:
        lasso_for_sel = LassoCV(
            alphas=np.logspace(-6, 2, 30 if quick else 50),
            cv=3, max_iter=10000, n_jobs=1,
        )
        sel_lasso = SelectFromModel(lasso_for_sel, threshold="median")
        pipe_sel_lasso_xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", sel_lasso),
            ("est", XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )),
        ])
        if quick:
            param_dist_lxgb = {
                "est__n_estimators": randint(100, 300),
                "est__max_depth": randint(3, 7),
                "est__learning_rate": loguniform(0.03, 0.2),
            }
        else:
            param_dist_lxgb = {
                "est__n_estimators": randint(50, 500),
                "est__max_depth": randint(3, 10),
                "est__learning_rate": loguniform(0.005, 0.3),
                "est__subsample": uniform(0.6, 0.4),
                "est__colsample_bytree": uniform(0.6, 0.4),
            }
        candidates.append(("SelLasso_XGB", pipe_sel_lasso_xgb, param_dist_lxgb))

    # 5. ARD-selector → RandomForest
    if _RF_AVAILABLE:
        sel_ard = SelectFromModel(
            ARDRegression(max_iter=200), threshold="median",
        )
        pipe_sel_ard_rf = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", sel_ard),
            ("est", RandomForestRegressor(random_state=42, n_jobs=1)),
        ])
        if quick:
            param_dist_arf = {
                "est__n_estimators": randint(100, 300),
                "est__max_depth": [None, 10, 20],
            }
        else:
            param_dist_arf = {
                "est__n_estimators": randint(100, 600),
                "est__max_depth": [None, 10, 20, 40],
                "est__min_samples_split": randint(2, 10),
                "est__min_samples_leaf": randint(1, 5),
            }
        candidates.append(("SelARD_RF", pipe_sel_ard_rf, param_dist_arf))

    # 6. Ridge-selector → XGBoost
    if _XGB_AVAILABLE:
        ridge_for_sel = RidgeCV(alphas=np.logspace(-3, 3, 20))
        sel_ridge = SelectFromModel(ridge_for_sel, threshold="median")
        pipe_sel_ridge_xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", sel_ridge),
            ("est", XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )),
        ])
        if quick:
            param_dist_rxgb = {
                "est__n_estimators": randint(100, 300),
                "est__max_depth": randint(3, 7),
            }
        else:
            param_dist_rxgb = {
                "est__n_estimators": randint(50, 500),
                "est__max_depth": randint(3, 10),
                "est__learning_rate": loguniform(0.005, 0.3),
            }
        candidates.append(("SelRidge_XGB", pipe_sel_ridge_xgb, param_dist_rxgb))

    # 7. Plain RandomForest (no selector)
    if _RF_AVAILABLE:
        pipe_rf = Pipeline([
            ("est", RandomForestRegressor(random_state=42, n_jobs=1)),
        ])
        if quick:
            param_dist_rf = {
                "est__n_estimators": randint(100, 300),
                "est__max_depth": [None, 10, 20],
            }
        else:
            param_dist_rf = {
                "est__n_estimators": randint(100, 1000),
                "est__max_depth": [None, 10, 20, 40],
                "est__min_samples_split": randint(2, 15),
                "est__min_samples_leaf": randint(1, 6),
                "est__max_features": ["sqrt", "log2", None],
            }
        candidates.append(("RF", pipe_rf, param_dist_rf))

    # 8. Plain XGBoost (no selector)
    if _XGB_AVAILABLE:
        pipe_xgb = Pipeline([
            ("est", XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )),
        ])
        if quick:
            param_dist_xgb = {
                "est__n_estimators": randint(100, 300),
                "est__max_depth": randint(3, 7),
                "est__learning_rate": loguniform(0.03, 0.2),
            }
        else:
            param_dist_xgb = {
                "est__n_estimators": randint(50, 1000),
                "est__max_depth": randint(3, 13),
                "est__learning_rate": loguniform(0.005, 0.3),
                "est__subsample": uniform(0.5, 0.5),
                "est__colsample_bytree": uniform(0.5, 0.5),
                "est__min_child_weight": randint(1, 8),
            }
        candidates.append(("XGB", pipe_xgb, param_dist_xgb))

    return candidates


# ---------------------------------------------------------------------------
# D. Nested CV evaluation
# ---------------------------------------------------------------------------

def nested_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: List[Tuple[str, Pipeline, Optional[Dict[str, Any]]]],
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, CandidateResult]:
    """Run nested cross-validation for all candidate pipelines.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Target variable.
    candidates : list of (name, pipeline, param_dist)
        Candidate pipelines.  If param_dist is None, the pipeline is
        fit directly (e.g. LassoCV selects alpha internally).
    n_outer : int
        Number of outer CV folds.
    n_inner : int
        Number of inner CV folds for HPO.
    n_iter : int
        Number of RandomizedSearchCV iterations.
    random_state : int
        Random seed.
    progress_callback : callable, optional
        Called with status messages.

    Returns
    -------
    results : dict of {name: CandidateResult}
    """
    X_arr = safe_array(X)
    y_arr = safe_array(y).ravel()

    strat_labels = make_stratify_labels(y_arr, n_bins=min(5, n_outer))
    outer_cv = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state,
    )

    results: Dict[str, CandidateResult] = {
        name: CandidateResult(name=name) for name, _, _ in candidates
    }

    feature_names = list(X.columns)

    for fold_i, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X_arr, strat_labels)
    ):
        X_tr = pd.DataFrame(X_arr[outer_train_idx], columns=feature_names)
        X_te = pd.DataFrame(X_arr[outer_test_idx], columns=feature_names)
        y_tr = pd.Series(y_arr[outer_train_idx])
        y_te = pd.Series(y_arr[outer_test_idx])

        inner_labels = make_stratify_labels(y_tr.values, n_bins=min(5, n_inner))
        _skf = StratifiedKFold(
            n_splits=n_inner, shuffle=True, random_state=random_state,
        )
        # Pre-compute splits using discretised labels so that
        # RandomizedSearchCV never calls split(X, y_continuous).
        inner_cv = list(_skf.split(safe_array(X_tr), inner_labels))

        for name, pipeline, param_dist in candidates:
            msg = f"Fold {fold_i + 1}/{n_outer}: {name}"
            if progress_callback:
                progress_callback(msg)
            logger.info("Nested CV: %s", msg)

            try:
                # Clone pipeline for this fold to avoid state leakage
                from sklearn.base import clone
                pipe_clone = clone(pipeline)

                if param_dist:
                    search = RandomizedSearchCV(
                        pipe_clone,
                        param_distributions=param_dist,
                        n_iter=min(n_iter, _count_param_combinations(param_dist)),
                        cv=inner_cv,
                        scoring="neg_root_mean_squared_error",
                        random_state=random_state,
                        n_jobs=get_safe_n_jobs(),
                        error_score=np.nan,
                    )
                    search.fit(
                        safe_array(X_tr), safe_array(y_tr).ravel(),
                    )
                    best = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipe_clone.fit(
                        safe_array(X_tr), safe_array(y_tr).ravel(),
                    )
                    best = pipe_clone
                    best_params = None

                preds = best.predict(safe_array(X_te))
                rmse = float(np.sqrt(mean_squared_error(
                    safe_array(y_te).ravel(), preds,
                )))

                results[name].outer_scores.append(rmse)
                results[name].best_params_per_fold.append(best_params)
                results[name].models.append(best)

            except Exception:
                logger.exception("Nested CV failed: %s fold %d", name, fold_i)
                results[name].outer_scores.append(float("nan"))
                results[name].best_params_per_fold.append(None)
                results[name].models.append(None)

    # Compute summary statistics
    for name, cr in results.items():
        scores = [s for s in cr.outer_scores if np.isfinite(s)]
        if scores:
            cr.mean_rmse = float(np.mean(scores))
            cr.std_rmse = float(np.std(scores))
        else:
            cr.mean_rmse = float("inf")
            cr.std_rmse = float("inf")

        # Compute median selected features across outer folds
        n_features_list: List[int] = []
        for est in cr.models:
            if est is None:
                continue
            if hasattr(est, "named_steps") and "selector" in est.named_steps:
                sel = est.named_steps["selector"]
                try:
                    n_features_list.append(int(sel.get_support().sum()))
                except Exception:
                    pass
            elif hasattr(est, "named_steps") and "est" in est.named_steps:
                # For Lasso: count non-zero coefficients
                inner_est = est.named_steps["est"]
                if hasattr(inner_est, "coef_"):
                    n_nz = int(np.sum(np.abs(inner_est.coef_) > 1e-10))
                    n_features_list.append(n_nz)

        if n_features_list:
            cr.median_n_features = int(np.median(n_features_list))

    return results


def _count_param_combinations(param_dist: Dict[str, Any]) -> int:
    """Estimate total combinations (for n_iter capping).

    Discrete lists are counted exactly; continuous distributions
    contribute a fixed budget of 10 each.
    """
    total = 1
    for values in param_dist.values():
        if isinstance(values, (list, tuple)):
            total *= len(values)
        else:
            total *= 10  # scipy.stats distributions: assume ~10 samples
    return total


# ---------------------------------------------------------------------------
# E. Model comparison & selection
# ---------------------------------------------------------------------------

def choose_best_model(
    results: Dict[str, CandidateResult],
    delta_rel: float = 0.03,
    prefer_sparser: bool = True,
) -> CandidateResult:
    """Choose the best model from nested CV results.

    Parameters
    ----------
    results : dict of {name: CandidateResult}
    delta_rel : float
        Relative RMSE threshold for considering models equivalent.
        If the second-best model is within ``delta_rel`` of the best
        and has fewer features, it is preferred.
    prefer_sparser : bool
        If True, prefer sparser models when RMSE is similar.

    Returns
    -------
    best : CandidateResult
    """
    valid = [
        cr for cr in results.values()
        if np.isfinite(cr.mean_rmse) and cr.mean_rmse < float("inf")
    ]
    if not valid:
        raise ValueError("No candidate produced valid results")

    sorted_candidates = sorted(valid, key=lambda c: c.mean_rmse)
    best = sorted_candidates[0]

    if prefer_sparser and len(sorted_candidates) > 1:
        for runner_up in sorted_candidates[1:]:
            if best.mean_rmse <= 0:
                break
            rel_diff = (runner_up.mean_rmse - best.mean_rmse) / best.mean_rmse
            if rel_diff < delta_rel:
                # Within tolerance — prefer sparser
                if (
                    best.median_n_features is not None
                    and runner_up.median_n_features is not None
                    and runner_up.median_n_features < best.median_n_features * 0.7
                ):
                    logger.info(
                        "Model selection: preferring sparser %s "
                        "(%d features) over %s (%d features), "
                        "RMSE diff %.4f < %.1f%%",
                        runner_up.name, runner_up.median_n_features,
                        best.name, best.median_n_features,
                        runner_up.mean_rmse - best.mean_rmse,
                        delta_rel * 100,
                    )
                    best = runner_up
                    break
            else:
                break  # No more candidates within tolerance

    logger.info(
        "Model selection: best=%s, RMSE=%.4f±%.4f, features=%s",
        best.name, best.mean_rmse, best.std_rmse,
        best.median_n_features,
    )
    return best


# ---------------------------------------------------------------------------
# F. Final refit & save
# ---------------------------------------------------------------------------

def refit_and_save(
    best_candidate: CandidateResult,
    pipeline_template: Pipeline,
    param_dist: Optional[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    random_state: int = 42,
) -> Tuple[Optional[Path], Optional[Path], Optional[List[str]]]:
    """Refit best pipeline on full data and save model + metadata.

    Parameters
    ----------
    best_candidate : CandidateResult
        The chosen candidate from ``choose_best_model``.
    pipeline_template : Pipeline
        The unfitted pipeline template for the best candidate.
    param_dist : dict or None
        Param distribution (used to get best params).
    X : DataFrame
        Full feature matrix.
    y : Series
        Full target.
    out_dir : Path
        Directory to save model and metadata.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    (model_path, meta_path, selected_features)
    """
    import joblib
    from sklearn.base import clone

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the best params from the most successful outer fold
    best_params = None
    best_model_from_folds = None
    best_fold_rmse = float("inf")
    for i, (score, params, model) in enumerate(zip(
        best_candidate.outer_scores,
        best_candidate.best_params_per_fold,
        best_candidate.models,
    )):
        if np.isfinite(score) and score < best_fold_rmse and model is not None:
            best_fold_rmse = score
            best_params = params
            best_model_from_folds = model

    # Clone and refit on full data
    pipe = clone(pipeline_template)

    if best_params:
        try:
            pipe.set_params(**best_params)
        except Exception:
            logger.warning(
                "Failed to set best_params for %s — using defaults",
                best_candidate.name,
            )
    elif best_params is None:
        # best_params is None (e.g. LassoCV selects alpha internally).
        # This is expected for CV-based estimators.  For HPO-based
        # candidates (XGB, RF) it means all folds failed — log a warning.
        if param_dist is not None:
            logger.warning(
                "No best_params found for %s (all folds may have failed) "
                "— refitting with default hyperparameters",
                best_candidate.name,
            )

    X_arr = safe_array(X)
    y_arr = safe_array(y).ravel()
    pipe.fit(X_arr, y_arr)

    # Extract selected features
    selected_features: Optional[List[str]] = None
    feature_names = list(X.columns)
    if hasattr(pipe, "named_steps") and "selector" in pipe.named_steps:
        sel = pipe.named_steps["selector"]
        try:
            support = sel.get_support()
            selected_features = [
                feature_names[i] for i, s in enumerate(support) if s
            ]
        except Exception:
            pass
    elif hasattr(pipe, "named_steps") and "est" in pipe.named_steps:
        inner_est = pipe.named_steps["est"]
        if hasattr(inner_est, "coef_"):
            # Lasso/ARD: features with non-zero coefficients
            coefs = np.abs(inner_est.coef_)
            # After scaler, feature order matches original
            if len(coefs) == len(feature_names):
                mask = coefs > 1e-10
                selected_features = [
                    feature_names[i] for i, s in enumerate(mask) if s
                ]

    # Save model
    model_path = out_dir / "best_model.joblib"
    joblib.dump(pipe, model_path, compress=3)

    # Save metadata
    meta = {
        "model_name": best_candidate.name,
        "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
        "outer_mean_rmse": round(best_candidate.mean_rmse, 6),
        "outer_std_rmse": round(best_candidate.std_rmse, 6),
        "outer_scores": [
            round(s, 6) if np.isfinite(s) else None
            for s in best_candidate.outer_scores
        ],
        "hyperparameters": _serialize_params(best_params),
        "n_features_input": len(feature_names),
        "n_features_selected": len(selected_features) if selected_features else None,
        "selected_features": selected_features,
        "median_n_features_cv": best_candidate.median_n_features,
    }
    meta_path = out_dir / "best_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(
        "Model saved: %s → %s (meta: %s)",
        best_candidate.name, model_path, meta_path,
    )
    return model_path, meta_path, selected_features


def _serialize_params(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Make param dict JSON-serialisable."""
    if params is None:
        return None
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# G. High-level API: run_model_selection
# ---------------------------------------------------------------------------

def run_model_selection(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Optional[Path] = None,
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 20,
    quick: bool = False,
    random_state: int = 42,
    delta_rel: float = 0.03,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ModelSelectionResult:
    """Run the complete model selection pipeline.

    1. Build candidate pipelines
    2. Run nested CV
    3. Choose best model (with sparsity preference)
    4. Refit on full data and save

    Parameters
    ----------
    X : DataFrame
        Feature matrix (numeric, NaN-free).
    y : Series
        Target variable.
    out_dir : Path, optional
        Output directory.  If None, model is not saved.
    n_outer : int
        Outer CV folds (default 5).
    n_inner : int
        Inner CV folds (default 3).
    n_iter : int
        RandomizedSearchCV iterations (default 20).
    quick : bool
        Use reduced search spaces.
    random_state : int
        Random seed.
    delta_rel : float
        Relative RMSE threshold for sparsity preference.
    progress_callback : callable, optional
        Called with status messages.

    Returns
    -------
    ModelSelectionResult
    """
    t0 = time.time()

    if progress_callback:
        progress_callback("Building candidate pipelines...")

    n_features = X.shape[1]
    candidates = build_candidate_pipelines(n_features, quick=quick)
    logger.info(
        "Model selection: %d candidates, %d features, "
        "outer=%d inner=%d n_iter=%d",
        len(candidates), n_features, n_outer, n_inner, n_iter,
    )

    if progress_callback:
        progress_callback(
            f"Running nested CV ({n_outer} outer × {n_inner} inner folds, "
            f"{len(candidates)} candidates)..."
        )

    cv_results = nested_cv_evaluate(
        X, y, candidates,
        n_outer=n_outer,
        n_inner=n_inner,
        n_iter=n_iter,
        random_state=random_state,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("Selecting best model...")

    best = choose_best_model(cv_results, delta_rel=delta_rel)

    # Find the pipeline template for the best candidate
    best_pipeline = None
    best_param_dist = None
    for name, pipe, pdist in candidates:
        if name == best.name:
            best_pipeline = pipe
            best_param_dist = pdist
            break

    result = ModelSelectionResult(
        best_name=best.name,
        best_mean_rmse=best.mean_rmse,
        best_std_rmse=best.std_rmse,
        all_candidates=list(cv_results.values()),
        elapsed_sec=time.time() - t0,
    )

    # Best params: use the best fold's params
    for params in best.best_params_per_fold:
        if params is not None:
            result.best_params = _serialize_params(params)
            break

    # Refit & save
    if out_dir is not None and best_pipeline is not None:
        if progress_callback:
            progress_callback("Refitting best model on full data...")
        model_path, meta_path, sel_features = refit_and_save(
            best, best_pipeline, best_param_dist,
            X, y, out_dir, random_state=random_state,
        )
        result.model_path = str(model_path) if model_path else None
        result.meta_path = str(meta_path) if meta_path else None
        result.selected_features = sel_features
        result.n_features_selected = len(sel_features) if sel_features else None

    result.elapsed_sec = time.time() - t0
    logger.info(
        "Model selection complete: best=%s RMSE=%.4f±%.4f in %.1f sec",
        result.best_name, result.best_mean_rmse,
        result.best_std_rmse, result.elapsed_sec,
    )
    return result
