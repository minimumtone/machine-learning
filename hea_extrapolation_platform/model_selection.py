"""
Nested CV + Regularization-Based Feature Selection Module
Nested CV + 正則化ベース特徴量選択モジュール

Implements a rigorous model selection pipeline:
  1. Target-stratified splitting (StratifiedKFold on binned target)
  2. Nested cross-validation (outer CV for evaluation, inner CV for hyperparameter tuning)
  3. Regularization-based feature selection (Lasso/Ridge/ARD as SelectFromModel)
  4. Model comparison with sparsity preference tie-breaker
  5. Final refit on full data with metadata export

Candidate Pipelines:
  - Lasso-selector + XGBoost
  - Ridge-selector + XGBoost
  - ARD-selector + XGBoost
  - ARD-selector + RandomForest

References:
  - Varma & Simon (2006) "Bias in error estimation when using cross-validation
    for model selection" BMC Bioinformatics 7:91
  - Cawley & Talbot (2010) "On Over-fitting in Model Selection and Subsequent
    Selection Bias in Performance Evaluation" JMLR 11:2079-2107
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ARDRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost availability check (same pattern as workflows.py)
# ---------------------------------------------------------------------------

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning(
        "xgboost not installed -- nested CV XGBoost candidates disabled"
    )


# ---------------------------------------------------------------------------
# C-contiguous helper (reuse pattern from workflows.py)
# ---------------------------------------------------------------------------

def _safe_np(source: Any) -> np.ndarray:
    """Return a C-contiguous float64 array from DataFrame / Series / ndarray.

    Prevents SIGSEGV from pandas 3.0 F-contiguous arrays reaching BLAS.
    """
    if isinstance(source, pd.DataFrame):
        arr = source.to_numpy(dtype="float64", na_value=np.nan)
    elif isinstance(source, pd.Series):
        arr = source.to_numpy(dtype="float64")
    elif isinstance(source, np.ndarray):
        arr = np.array(source, dtype="float64")
    else:
        arr = np.asarray(source, dtype="float64")
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class NestedCVFoldResult:
    """Result of a single outer fold for a single candidate pipeline."""
    fold_idx: int
    rmse_test: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    n_selected_features: int = 0
    selected_feature_names: List[str] = field(default_factory=list)


@dataclass
class NestedCVModelResult:
    """Aggregated nested CV results for a single candidate pipeline."""
    name: str
    outer_scores: List[float] = field(default_factory=list)
    fold_results: List[NestedCVFoldResult] = field(default_factory=list)
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    median_n_selected: float = 0.0

    def compute_summary(self) -> None:
        """Compute summary statistics from outer_scores."""
        if self.outer_scores:
            self.mean_rmse = float(np.mean(self.outer_scores))
            self.std_rmse = float(np.std(self.outer_scores))
        if self.fold_results:
            n_sel = [fr.n_selected_features for fr in self.fold_results]
            self.median_n_selected = float(np.median(n_sel))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON export."""
        return {
            "name": self.name,
            "mean_rmse": round(self.mean_rmse, 6),
            "std_rmse": round(self.std_rmse, 6),
            "median_n_selected": self.median_n_selected,
            "outer_scores": [round(s, 6) for s in self.outer_scores],
            "fold_results": [
                {
                    "fold_idx": fr.fold_idx,
                    "rmse_test": round(fr.rmse_test, 6),
                    "n_selected_features": fr.n_selected_features,
                    "selected_feature_names": fr.selected_feature_names,
                    "best_params": {
                        k: _serializable_value(v)
                        for k, v in fr.best_params.items()
                    },
                }
                for fr in self.fold_results
            ],
        }


@dataclass
class NestedCVSummary:
    """Summary of nested CV model comparison."""
    model_results: List[NestedCVModelResult] = field(default_factory=list)
    best_model_name: str = ""
    best_mean_rmse: float = float("inf")
    best_selected_features: List[str] = field(default_factory=list)
    selection_reason: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_model_name": self.best_model_name,
            "best_mean_rmse": round(self.best_mean_rmse, 6),
            "best_selected_features": self.best_selected_features,
            "selection_reason": self.selection_reason,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "model_results": [mr.to_dict() for mr in self.model_results],
        }


def _serializable_value(v: Any) -> Any:
    """Convert a single value to JSON-serialisable form."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if v is None:
        return None
    return v


# ---------------------------------------------------------------------------
# A. make_stratify_labels — target binning for StratifiedKFold
# ---------------------------------------------------------------------------

def make_stratify_labels(
    y: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """Discretise continuous target *y* into integer labels for StratifiedKFold.

    Uses quantile-based binning (KBinsDiscretizer strategy='quantile').
    Automatically reduces ``n_bins`` if the sample count is too small.

    Parameters
    ----------
    y : 1-D array
        Continuous target values.
    n_bins : int
        Number of quantile bins (default 5).

    Returns
    -------
    labels : 1-D integer array of shape (len(y),) with values in [0, n_bins-1].
    """
    y_arr = _safe_np(y).ravel()
    n = len(y_arr)

    # Reduce bins if sample count is small
    actual_bins = min(n_bins, max(2, n // 3))
    if actual_bins < n_bins:
        logger.warning(
            "make_stratify_labels: reduced n_bins from %d to %d (n=%d)",
            n_bins, actual_bins, n,
        )

    try:
        kbd = KBinsDiscretizer(
            n_bins=actual_bins,
            encode="ordinal",
            strategy="quantile",
            subsample=None,
        )
        labels = kbd.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(int)
    except ValueError:
        # Fallback: if quantile binning fails (e.g. too many identical values),
        # use uniform binning
        logger.warning(
            "Quantile binning failed; falling back to uniform binning"
        )
        kbd = KBinsDiscretizer(
            n_bins=actual_bins,
            encode="ordinal",
            strategy="uniform",
            subsample=None,
        )
        labels = kbd.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(int)

    # Safety: ensure every bin has at least 1 sample
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        # Degenerate case — split in half
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1

    return labels


# ---------------------------------------------------------------------------
# B. Candidate pipeline definitions
# ---------------------------------------------------------------------------

@dataclass
class CandidatePipeline:
    """Definition of a candidate pipeline for nested CV."""
    name: str
    pipeline: Pipeline
    param_distributions: Dict[str, Any]


def _get_inner_n_jobs() -> int:
    """Return n_jobs for inner estimators. Respects HEA_INNER_N_JOBS env var."""
    import os
    raw = os.environ.get("HEA_INNER_N_JOBS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def get_candidate_pipelines(
    n_features: int = 10,
    random_state: int = 42,
) -> List[CandidatePipeline]:
    """Build candidate pipelines with their hyperparameter search spaces.

    Each pipeline follows: StandardScaler -> SelectFromModel(selector) -> Model

    Parameters
    ----------
    n_features : int
        Number of input features (used to set sensible defaults).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    List of CandidatePipeline objects.
    """
    inner_jobs = _get_inner_n_jobs()
    candidates: List[CandidatePipeline] = []

    # --- Lasso-selector + XGBoost ---
    if _XGB_AVAILABLE:
        candidates.append(CandidatePipeline(
            name="Lasso-XGB",
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectFromModel(
                    LassoCV(cv=3, max_iter=10000, random_state=random_state),
                    threshold="median",
                )),
                ("model", XGBRegressor(
                    random_state=random_state,
                    n_jobs=inner_jobs,
                    verbosity=0,
                )),
            ]),
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [3, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "selector__threshold": ["median", "mean", "0.5*mean"],
            },
        ))

    # --- Ridge-selector + XGBoost ---
    if _XGB_AVAILABLE:
        candidates.append(CandidatePipeline(
            name="Ridge-XGB",
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectFromModel(
                    RidgeCV(alphas=np.logspace(-3, 3, 20)),
                    threshold="median",
                )),
                ("model", XGBRegressor(
                    random_state=random_state,
                    n_jobs=inner_jobs,
                    verbosity=0,
                )),
            ]),
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [3, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "selector__threshold": ["median", "mean", "0.5*mean"],
            },
        ))

    # --- ARD-selector + XGBoost ---
    if _XGB_AVAILABLE:
        candidates.append(CandidatePipeline(
            name="ARD-XGB",
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectFromModel(
                    ARDRegression(max_iter=300),
                    threshold="median",
                )),
                ("model", XGBRegressor(
                    random_state=random_state,
                    n_jobs=inner_jobs,
                    verbosity=0,
                )),
            ]),
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [3, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "selector__threshold": ["median", "mean", "0.5*mean"],
            },
        ))

    # --- ARD-selector + RandomForest ---
    candidates.append(CandidatePipeline(
        name="ARD-RF",
        pipeline=Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(
                ARDRegression(max_iter=300),
                threshold="median",
            )),
            ("model", RandomForestRegressor(
                random_state=random_state,
                n_jobs=inner_jobs,
            )),
        ]),
        param_distributions={
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 6, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "selector__threshold": ["median", "mean", "0.5*mean"],
        },
    ))

    logger.info(
        "Built %d candidate pipelines: %s",
        len(candidates),
        [c.name for c in candidates],
    )
    return candidates


# ---------------------------------------------------------------------------
# C. nested_cv_evaluate — main nested CV loop
# ---------------------------------------------------------------------------

def nested_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    candidate_pipelines: Optional[List[CandidatePipeline]] = None,
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
    n_jobs: int = 1,
    progress_callback: Optional[Any] = None,
) -> Dict[str, NestedCVModelResult]:
    """Run nested cross-validation for model selection.

    Outer loop: StratifiedKFold(n_splits=n_outer) on binned target.
    Inner loop: RandomizedSearchCV(cv=StratifiedKFold(n_splits=n_inner)).

    Parameters
    ----------
    X : 2-D array, shape (n_samples, n_features)
        Feature matrix (must be C-contiguous float64).
    y : 1-D array, shape (n_samples,)
        Target values.
    feature_names : list of str
        Feature column names for reporting selected features.
    candidate_pipelines : list of CandidatePipeline, optional
        If None, uses default candidates from get_candidate_pipelines().
    n_outer : int
        Number of outer CV folds (default 5).
    n_inner : int
        Number of inner CV folds (default 3).
    n_iter : int
        Number of RandomizedSearchCV iterations (default 20).
    random_state : int
        Random seed.
    n_jobs : int
        Parallelism for RandomizedSearchCV (default 1 to avoid contention).
    progress_callback : callable, optional
        Called with (current_step, total_steps, message) for progress reporting.

    Returns
    -------
    dict of {pipeline_name: NestedCVModelResult}
    """
    X_arr = _safe_np(X)
    y_arr = _safe_np(y).ravel()

    if candidate_pipelines is None:
        candidate_pipelines = get_candidate_pipelines(
            n_features=X_arr.shape[1],
            random_state=random_state,
        )

    # Create stratification labels from target
    strat_labels = make_stratify_labels(y_arr, n_bins=min(5, n_outer))

    # Outer CV splitter
    outer_cv = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state,
    )

    # Inner CV splitter (for RandomizedSearchCV)
    inner_cv = StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=random_state,
    )

    results: Dict[str, NestedCVModelResult] = {}
    total_steps = len(candidate_pipelines) * n_outer
    current_step = 0

    for cand in candidate_pipelines:
        model_result = NestedCVModelResult(name=cand.name)

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X_arr, strat_labels)
        ):
            current_step += 1
            if progress_callback is not None:
                try:
                    progress_callback(
                        current_step, total_steps,
                        f"Nested CV: {cand.name} fold {fold_idx + 1}/{n_outer}",
                    )
                except Exception:
                    pass

            X_train = np.ascontiguousarray(X_arr[train_idx])
            X_test = np.ascontiguousarray(X_arr[test_idx])
            y_train = np.ascontiguousarray(y_arr[train_idx])
            y_test = np.ascontiguousarray(y_arr[test_idx])

            # Inner CV labels for stratification
            inner_labels = make_stratify_labels(y_train, n_bins=min(5, n_inner))

            # Clone pipeline for this fold (sklearn's clone is called by
            # RandomizedSearchCV internally)
            from sklearn.base import clone
            pipe_clone = clone(cand.pipeline)

            # Inner RandomizedSearchCV
            try:
                search = RandomizedSearchCV(
                    estimator=pipe_clone,
                    param_distributions=cand.param_distributions,
                    n_iter=min(n_iter, _count_param_combinations(cand.param_distributions)),
                    cv=inner_cv.split(X_train, inner_labels),
                    scoring="neg_root_mean_squared_error",
                    n_jobs=n_jobs,
                    random_state=random_state,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_train, y_train)

                # Evaluate on outer test fold
                y_pred = search.best_estimator_.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

                # Extract selected features
                selected_names = _extract_selected_features(
                    search.best_estimator_, feature_names,
                )

                fold_result = NestedCVFoldResult(
                    fold_idx=fold_idx,
                    rmse_test=rmse,
                    best_params=dict(search.best_params_),
                    n_selected_features=len(selected_names),
                    selected_feature_names=selected_names,
                )

            except Exception as exc:
                logger.warning(
                    "Nested CV: %s fold %d failed: %s",
                    cand.name, fold_idx, exc,
                )
                fold_result = NestedCVFoldResult(
                    fold_idx=fold_idx,
                    rmse_test=float("inf"),
                )

            model_result.outer_scores.append(fold_result.rmse_test)
            model_result.fold_results.append(fold_result)

        model_result.compute_summary()
        results[cand.name] = model_result

        logger.info(
            "Nested CV [%s]: mean_RMSE=%.4f +/- %.4f, "
            "median_selected=%d features",
            cand.name,
            model_result.mean_rmse,
            model_result.std_rmse,
            int(model_result.median_n_selected),
        )

    return results


def _count_param_combinations(param_distributions: Dict[str, Any]) -> int:
    """Count total number of parameter combinations."""
    total = 1
    for values in param_distributions.values():
        if isinstance(values, (list, tuple)):
            total *= len(values)
        else:
            total *= 10  # default for distributions
    return total


def _extract_selected_features(
    fitted_pipeline: Pipeline,
    feature_names: List[str],
) -> List[str]:
    """Extract the names of features selected by the selector step.

    Parameters
    ----------
    fitted_pipeline : sklearn Pipeline
        A fitted pipeline containing a 'selector' step (SelectFromModel).
    feature_names : list of str
        Original feature names before selection.

    Returns
    -------
    List of selected feature names.
    """
    try:
        selector = fitted_pipeline.named_steps.get("selector")
        if selector is None:
            return list(feature_names)

        support_mask = selector.get_support()
        selected = [
            name for name, sel in zip(feature_names, support_mask) if sel
        ]
        return selected
    except Exception as exc:
        logger.warning("Could not extract selected features: %s", exc)
        return list(feature_names)


# ---------------------------------------------------------------------------
# D. choose_best_model_from_results — model comparison with sparsity preference
# ---------------------------------------------------------------------------

def choose_best_model_from_results(
    results: Dict[str, NestedCVModelResult],
    delta_rel: float = 0.03,
    prefer_sparser: bool = True,
) -> Tuple[str, NestedCVModelResult, str]:
    """Choose the best model from nested CV results.

    Primary criterion: minimum outer CV mean RMSE.
    Tie-breaker (when ``prefer_sparser=True``): if the 2nd-best model's
    mean RMSE is within ``delta_rel`` (3%) of the best, and it uses 30%
    fewer features, prefer the sparser model.

    Parameters
    ----------
    results : dict of {name: NestedCVModelResult}
    delta_rel : float
        Relative RMSE tolerance for tie-breaking (default 0.03 = 3%).
    prefer_sparser : bool
        Whether to prefer sparser models in ties (default True).

    Returns
    -------
    (best_name, best_result, reason)
    """
    if not results:
        raise ValueError("No nested CV results to compare")

    # Filter out models with infinite RMSE
    valid = {
        name: r for name, r in results.items()
        if np.isfinite(r.mean_rmse)
    }
    if not valid:
        # All failed — return first
        first_name = next(iter(results))
        return first_name, results[first_name], "all models failed"

    # Sort by mean RMSE ascending
    sorted_models = sorted(valid.items(), key=lambda kv: kv[1].mean_rmse)

    best_name, best_result = sorted_models[0]
    reason = f"lowest mean RMSE = {best_result.mean_rmse:.4f}"

    # Tie-breaker: sparsity preference
    if prefer_sparser and len(sorted_models) >= 2:
        second_name, second_result = sorted_models[1]
        rmse_diff_rel = abs(
            second_result.mean_rmse - best_result.mean_rmse
        ) / max(best_result.mean_rmse, 1e-10)

        if rmse_diff_rel < delta_rel:
            # Within tolerance — check sparsity
            best_n_feat = best_result.median_n_selected
            second_n_feat = second_result.median_n_selected

            if best_n_feat > 0 and second_n_feat > 0:
                sparsity_ratio = second_n_feat / best_n_feat
                if sparsity_ratio < 0.70:
                    # Second model uses 30%+ fewer features
                    logger.info(
                        "Sparsity preference: %s (%.1f features) over "
                        "%s (%.1f features) — RMSE diff %.2f%% < %.0f%%",
                        second_name, second_n_feat,
                        best_name, best_n_feat,
                        rmse_diff_rel * 100, delta_rel * 100,
                    )
                    best_name = second_name
                    best_result = second_result
                    reason = (
                        f"sparsity preference: {second_name} "
                        f"({second_n_feat:.0f} features) chosen over "
                        f"{sorted_models[0][0]} ({best_n_feat:.0f} features) "
                        f"— RMSE diff {rmse_diff_rel:.2%} < {delta_rel:.0%}"
                    )

    logger.info("Best model: %s — %s", best_name, reason)
    return best_name, best_result, reason


# ---------------------------------------------------------------------------
# E. refit_and_save_best_model — final refit + save
# ---------------------------------------------------------------------------

def refit_and_save_best_model(
    best_pipeline: Pipeline,
    best_params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    model_result: Optional[NestedCVModelResult] = None,
    selection_reason: str = "",
) -> Tuple[Path, Path]:
    """Refit the best pipeline on full data and save model + metadata.

    Parameters
    ----------
    best_pipeline : Pipeline
        The pipeline template (will be cloned and set with best_params).
    best_params : dict
        Best hyperparameters from inner CV.
    X : 2-D array
        Full feature matrix.
    y : 1-D array
        Full target vector.
    feature_names : list of str
        Feature column names.
    out_dir : Path
        Directory to save model and metadata.
    model_result : NestedCVModelResult, optional
        Nested CV result for metadata.
    selection_reason : str
        Why this model was selected.

    Returns
    -------
    (model_path, meta_path) : tuple of Path
    """
    import joblib
    from sklearn.base import clone

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_arr = _safe_np(X)
    y_arr = _safe_np(y).ravel()

    # Clone and set best params
    pipe = clone(best_pipeline)
    pipe.set_params(**best_params)

    logger.info("Refitting best model on full dataset (n=%d)", len(y_arr))
    pipe.fit(X_arr, y_arr)

    # Extract selected features after full refit
    selected_features = _extract_selected_features(pipe, feature_names)

    # Save model
    model_path = out_dir / "best_model.joblib"
    joblib.dump(pipe, model_path, compress=3)
    logger.info("Saved best model to %s", model_path)

    # Save metadata
    meta = {
        "model_name": model_result.name if model_result else "unknown",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "selection_reason": selection_reason,
        "n_samples": len(y_arr),
        "n_features_input": X_arr.shape[1],
        "n_features_selected": len(selected_features),
        "selected_features": selected_features,
        "best_params": {
            k: _serializable_value(v)
            for k, v in best_params.items()
        },
    }
    if model_result is not None:
        meta["outer_mean_rmse"] = round(model_result.mean_rmse, 6)
        meta["outer_std_rmse"] = round(model_result.std_rmse, 6)
        meta["median_n_selected"] = model_result.median_n_selected

    meta_path = out_dir / "best_model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved model metadata to %s", meta_path)

    return model_path, meta_path


# ---------------------------------------------------------------------------
# F. run_nested_cv — high-level orchestrator
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: Any,
    y: Any,
    feature_names: List[str],
    out_dir: Optional[Path] = None,
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
    delta_rel: float = 0.03,
    progress_callback: Optional[Any] = None,
) -> NestedCVSummary:
    """High-level function: run nested CV, select best model, optionally save.

    This is the main entry point called by runner.py and the GUI.

    Parameters
    ----------
    X : DataFrame or 2-D array
        Feature matrix.
    y : Series or 1-D array
        Target vector.
    feature_names : list of str
        Feature column names.
    out_dir : Path, optional
        If provided, the best model is refitted on full data and saved.
    n_outer : int
        Outer CV folds (default 5).
    n_inner : int
        Inner CV folds (default 3).
    n_iter : int
        RandomizedSearchCV iterations (default 20).
    random_state : int
        Random seed (default 42).
    delta_rel : float
        Sparsity tie-breaker tolerance (default 0.03).
    progress_callback : callable, optional
        Progress callback.

    Returns
    -------
    NestedCVSummary with all results and best model info.
    """
    t_start = time.time()

    X_arr = _safe_np(X)
    y_arr = _safe_np(y).ravel()

    logger.info(
        "Starting nested CV: n=%d, p=%d, outer=%d, inner=%d, n_iter=%d",
        len(y_arr), X_arr.shape[1], n_outer, n_inner, n_iter,
    )

    # Build candidates
    candidates = get_candidate_pipelines(
        n_features=X_arr.shape[1],
        random_state=random_state,
    )

    # Run nested CV
    results = nested_cv_evaluate(
        X=X_arr,
        y=y_arr,
        feature_names=feature_names,
        candidate_pipelines=candidates,
        n_outer=n_outer,
        n_inner=n_inner,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=_get_inner_n_jobs(),
        progress_callback=progress_callback,
    )

    # Choose best model
    best_name, best_result, reason = choose_best_model_from_results(
        results, delta_rel=delta_rel,
    )

    # Collect best selected features from all folds
    best_features_all: List[str] = []
    for fr in best_result.fold_results:
        best_features_all.extend(fr.selected_feature_names)
    # Most frequently selected features across folds
    from collections import Counter
    feat_counts = Counter(best_features_all)
    # Features selected in majority of folds
    majority_threshold = max(1, n_outer // 2)
    consensus_features = [
        feat for feat, count in feat_counts.most_common()
        if count >= majority_threshold
    ]

    summary = NestedCVSummary(
        model_results=list(results.values()),
        best_model_name=best_name,
        best_mean_rmse=best_result.mean_rmse,
        best_selected_features=consensus_features,
        selection_reason=reason,
        elapsed_sec=time.time() - t_start,
    )

    # Optionally save best model
    if out_dir is not None:
        # Find best params from last fold (or median fold)
        best_params = {}
        if best_result.fold_results:
            # Use params from fold with median RMSE
            fold_rmses = [fr.rmse_test for fr in best_result.fold_results]
            median_idx = int(np.argsort(fold_rmses)[len(fold_rmses) // 2])
            best_params = best_result.fold_results[median_idx].best_params

        # Find the original pipeline template
        best_candidate = None
        for cand in candidates:
            if cand.name == best_name:
                best_candidate = cand
                break

        if best_candidate is not None and best_params:
            refit_and_save_best_model(
                best_pipeline=best_candidate.pipeline,
                best_params=best_params,
                X=X_arr,
                y=y_arr,
                feature_names=feature_names,
                out_dir=out_dir,
                model_result=best_result,
                selection_reason=reason,
            )

        # Save nested CV results JSON
        results_path = out_dir / "nested_cv_results.json"
        results_path.write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved nested CV results to %s", results_path)

    logger.info(
        "Nested CV complete: best=%s (RMSE=%.4f), %d features, %.1f sec",
        best_name, best_result.mean_rmse,
        len(consensus_features), summary.elapsed_sec,
    )

    return summary
