"""
ML Workflow Module for HEA Extrapolation Platform
MLワークフローモジュール

Three workflow templates:
  WF-LIN  - Linear regression (coefficient analysis, residual diagnostics)
  WF-XGB  - XGBoost with hyperparameter optimisation
  WF-ENS  - Seed-varied ensemble for uncertainty quantification
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression, Lasso, LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


from hea_extrapolation_platform._utils import safe_array as _safe_np  # noqa: E402


def _get_inner_n_jobs() -> int:
    """Return the n_jobs for inner estimators (GridSearchCV, RandomForest, etc.).

    Problem (#11): When ``RandomizedSearchCV(n_jobs=-1)`` runs inside a
    ``ProcessPoolExecutor`` worker, both compete for CPU cores.  The inner
    parallelism should be limited to 1 per worker to avoid resource contention.

    Respects ``HEA_INNER_N_JOBS`` environment variable.  Defaults to 1.
    """
    import os
    raw = os.environ.get("HEA_INNER_N_JOBS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1

try:
    from xgboost import XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed – WF-XGB will fall back to GradientBoosting")


# ---------------------------------------------------------------------------
# Run result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Container for a single workflow run result (MLflow-style)."""

    workflow: str
    feature_set: str
    split_policy: str
    seed: int
    fold: int

    # Metrics
    rmse_train: float = 0.0
    rmse_test: float = 0.0
    mae_train: float = 0.0
    mae_test: float = 0.0
    r2_train: float = 0.0
    r2_test: float = 0.0

    # Predictions (test set)
    y_test_true: Optional[np.ndarray] = None
    y_test_pred: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None

    # Extra artefacts
    params: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def __post_init__(self) -> None:
        """Enforce plain Python floats and C-contiguous numpy arrays.

        pandas 3.0 produces numpy scalars that retain references to
        F-contiguous memory blocks.  Converting metrics to ``float()``
        severs these references.  numpy arrays stored in the result are
        forced C-contiguous to prevent downstream SIGSEGV.
        """
        # Sever numpy scalar references to F-contiguous memory
        self.rmse_train = float(self.rmse_train)
        self.rmse_test = float(self.rmse_test)
        self.mae_train = float(self.mae_train)
        self.mae_test = float(self.mae_test)
        self.r2_train = float(self.r2_train)
        self.r2_test = float(self.r2_test)
        self.elapsed_sec = float(self.elapsed_sec)
        # Ensure all stored numpy arrays are C-contiguous
        if self.y_test_true is not None:
            self.y_test_true = np.ascontiguousarray(self.y_test_true)
        if self.y_test_pred is not None:
            self.y_test_pred = np.ascontiguousarray(self.y_test_pred)
        if self.test_indices is not None:
            self.test_indices = np.ascontiguousarray(self.test_indices)

    def metrics_dict(self) -> Dict[str, float]:
        return {
            "rmse_train": self.rmse_train,
            "rmse_test": self.rmse_test,
            "mae_train": self.mae_train,
            "mae_test": self.mae_test,
            "r2_train": self.r2_train,
            "r2_test": self.r2_test,
        }


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R².

    When fewer than 2 samples are present, r2_score is undefined; we
    return NaN rather than letting sklearn emit an UndefinedMetricWarning
    and propagate NaN through subsequent computations.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Base workflow
# ---------------------------------------------------------------------------


class BaseWorkflow(ABC):
    """Abstract ML workflow."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        ...


# ---------------------------------------------------------------------------
# WF-LIN: Linear regression
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dimensionality-reduction helper
# ---------------------------------------------------------------------------


def _make_pca_step(
    n_features: int,
    dim_reduction: bool,
    variance_ratio: float = 0.95,
) -> List[Tuple[str, Any]]:
    """Return pipeline steps for optional PCA dimensionality reduction.

    When *dim_reduction* is True a PCA step is inserted **after** scaling
    that retains *variance_ratio* (default 95 %) of total variance.
    ``n_components`` is capped at ``min(n_samples, n_features)`` by sklearn
    automatically, so this is safe even for small datasets.

    Returns a list of ``(name, estimator)`` tuples ready for
    ``Pipeline(steps=[...])``.
    """
    if not dim_reduction or n_features <= 2:
        return []
    # Keep at most n_features components; PCA will further clamp to
    # min(n_samples, n_features) internally.
    return [("pca", PCA(n_components=variance_ratio, svd_solver="full"))]


class WorkflowLIN(BaseWorkflow):
    """Linear regression workflow.

    Purpose:
    - Feature sign validation
    - Monotonicity check
    - Leak detection via residual analysis

    Uses Ridge (alpha=1.0) for numerical stability.
    """

    name = "WF-LIN"

    def __init__(self, alpha: float = 1.0, dim_reduction: bool = True) -> None:
        self._alpha = alpha
        self._dim_reduction = dim_reduction

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        t0 = time.time()
        logger.debug("WF-LIN: train=%d, test=%d, features=%d",
                      len(X_train), len(X_test), X_train.shape[1])

        steps: List[Tuple[str, Any]] = [
            ("scaler", StandardScaler()),
            *_make_pca_step(X_train.shape[1], self._dim_reduction),
            ("model", Ridge(alpha=self._alpha)),
        ]
        pipe = Pipeline(steps)
        pipe.fit(_safe_np(X_train), _safe_np(y_train))

        y_train_pred = pipe.predict(_safe_np(X_train))
        y_test_pred = pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        # Extract standardised coefficients.
        # Ridge inside the Pipeline sees *already-standardised* X (via the
        # preceding StandardScaler step).  Therefore ``coef_raw[j]`` already
        # represents "change in y per 1-std change in X_j".  To express
        # the coefficient in units of "std_y per std_X", we only divide by
        # std_y — multiplying by std_X again would double-count scaling.
        model: Ridge = pipe.named_steps["model"]
        coef_raw = model.coef_
        # Use ddof=1 (sample std) and guard against single-sample or
        # constant-y edge cases where std would be 0.
        if len(y_train) > 1:
            std_y = float(np.std(_safe_np(y_train), ddof=1))
        else:
            std_y = 0.0
        if std_y < 1e-12:
            std_y = 1.0  # avoid division by zero; coefficients stay in raw units
        coef_std = coef_raw / std_y

        result = RunResult(
            workflow=self.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=train_s["rmse"],
            rmse_test=test_s["rmse"],
            mae_train=train_s["mae"],
            mae_test=test_s["mae"],
            r2_train=train_s["r2"],
            r2_test=test_s["r2"],
            y_test_true=_safe_np(y_test).copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params={"alpha": self._alpha, "dim_reduction": self._dim_reduction},
            artifacts={
                "coef_raw": (
                    dict(zip(X_train.columns, coef_raw.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(c) for i, c in enumerate(coef_raw)}
                ),
                "coef_std": (
                    dict(zip(X_train.columns, coef_std.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(c) for i, c in enumerate(coef_std)}
                ),
                "residuals_test": (_safe_np(y_test) - y_test_pred).tolist(),
                "n_components": (
                    pipe.named_steps["pca"].n_components_
                    if "pca" in pipe.named_steps else X_train.shape[1]
                ),
            },
            elapsed_sec=time.time() - t0,
        )
        return result


# ---------------------------------------------------------------------------
# WF-LASSO: Lasso regression (L1 regularisation)
# ---------------------------------------------------------------------------


class WorkflowLASSO(BaseWorkflow):
    """Lasso (L1) regression workflow.

    Purpose:
    - Sparse feature selection via L1 penalty
    - Identifies which features can be zeroed out
    - Complementary to Ridge (L2) for feature importance analysis

    Uses LassoCV to automatically select the best alpha.
    """

    name = "WF-LASSO"

    def __init__(self, dim_reduction: bool = True) -> None:
        self._dim_reduction = dim_reduction

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        t0 = time.time()
        logger.debug("WF-LASSO: train=%d, test=%d, features=%d",
                      len(X_train), len(X_test), X_train.shape[1])

        steps: List[Tuple[str, Any]] = [
            ("scaler", StandardScaler()),
            *_make_pca_step(X_train.shape[1], self._dim_reduction),
            ("model", LassoCV(cv=max(2, min(10, len(X_train))), random_state=seed, max_iter=10000)),
        ]
        pipe = Pipeline(steps)
        pipe.fit(_safe_np(X_train), _safe_np(y_train))

        y_train_pred = pipe.predict(_safe_np(X_train))
        y_test_pred = pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        model: LassoCV = pipe.named_steps["model"]
        coef_raw = model.coef_
        n_nonzero = int(np.sum(np.abs(coef_raw) > 1e-10))

        if len(y_train) > 1:
            std_y = float(np.std(_safe_np(y_train), ddof=1))
        else:
            std_y = 0.0
        if std_y < 1e-12:
            std_y = 1.0
        coef_std = coef_raw / std_y

        result = RunResult(
            workflow=self.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=train_s["rmse"],
            rmse_test=test_s["rmse"],
            mae_train=train_s["mae"],
            mae_test=test_s["mae"],
            r2_train=train_s["r2"],
            r2_test=test_s["r2"],
            y_test_true=_safe_np(y_test).copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params={"alpha": float(model.alpha_), "dim_reduction": self._dim_reduction},
            artifacts={
                "coef_raw": (
                    dict(zip(X_train.columns, coef_raw.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(c) for i, c in enumerate(coef_raw)}
                ),
                "coef_std": (
                    dict(zip(X_train.columns, coef_std.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(c) for i, c in enumerate(coef_std)}
                ),
                "n_nonzero_features": n_nonzero,
                "residuals_test": (_safe_np(y_test) - y_test_pred).tolist(),
                "n_components": (
                    pipe.named_steps["pca"].n_components_
                    if "pca" in pipe.named_steps else X_train.shape[1]
                ),
            },
            elapsed_sec=time.time() - t0,
        )
        return result


# ---------------------------------------------------------------------------
# WF-ARD: Automatic Relevance Determination (Bayesian sparse regression)
# ---------------------------------------------------------------------------


class WorkflowARD(BaseWorkflow):
    """ARD (Automatic Relevance Determination) Bayesian regression workflow.

    Purpose:
    - Bayesian sparse feature importance estimation
    - Automatic pruning of irrelevant features via per-feature precision
    - Uncertainty-aware coefficient estimation
    """

    name = "WF-ARD"

    def __init__(self, dim_reduction: bool = True) -> None:
        self._dim_reduction = dim_reduction

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        t0 = time.time()
        logger.debug("WF-ARD: train=%d, test=%d, features=%d",
                      len(X_train), len(X_test), X_train.shape[1])

        steps: List[Tuple[str, Any]] = [
            ("scaler", StandardScaler()),
            *_make_pca_step(X_train.shape[1], self._dim_reduction),
            ("model", ARDRegression(max_iter=500)),
        ]
        pipe = Pipeline(steps)
        pipe.fit(_safe_np(X_train), _safe_np(y_train))

        y_train_pred = pipe.predict(_safe_np(X_train))
        y_test_pred = pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        model: ARDRegression = pipe.named_steps["model"]
        coef_raw = model.coef_
        # ARD's lambda_ gives per-feature precision (inverse variance)
        # Higher lambda_ = less relevant feature
        relevance = 1.0 / (model.lambda_ + 1e-10)
        relevance_norm = relevance / relevance.max() if relevance.max() > 0 else relevance

        if len(y_train) > 1:
            std_y = float(np.std(_safe_np(y_train), ddof=1))
        else:
            std_y = 0.0
        if std_y < 1e-12:
            std_y = 1.0
        coef_std = coef_raw / std_y

        result = RunResult(
            workflow=self.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=train_s["rmse"],
            rmse_test=test_s["rmse"],
            mae_train=train_s["mae"],
            mae_test=test_s["mae"],
            r2_train=train_s["r2"],
            r2_test=test_s["r2"],
            y_test_true=_safe_np(y_test).copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params={"dim_reduction": self._dim_reduction},
            artifacts={
                "coef_raw": (
                    dict(zip(X_train.columns, coef_raw.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(c) for i, c in enumerate(coef_raw)}
                ),
                "coef_std": (
                    dict(zip(X_train.columns, coef_std.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(c) for i, c in enumerate(coef_std)}
                ),
                "relevance_scores": (
                    dict(zip(X_train.columns, relevance_norm.tolist()))
                    if "pca" not in pipe.named_steps
                    else {f"PC{i}": float(r) for i, r in enumerate(relevance_norm)}
                ),
                "residuals_test": (_safe_np(y_test) - y_test_pred).tolist(),
                "n_components": (
                    pipe.named_steps["pca"].n_components_
                    if "pca" in pipe.named_steps else X_train.shape[1]
                ),
            },
            elapsed_sec=time.time() - t0,
        )
        return result


# ---------------------------------------------------------------------------
# WF-XGB: XGBoost + HPO
# ---------------------------------------------------------------------------


class WorkflowXGB(BaseWorkflow):
    """XGBoost workflow with grid-search hyperparameter optimisation.

    Purpose:
    - Capture non-linear interactions
    - Permutation importance
    """

    name = "WF-XGB"

    def __init__(self, n_cv: int = 10, quick: bool = False, dim_reduction: bool = True) -> None:
        self._n_cv = n_cv
        self._quick = quick
        self._dim_reduction = dim_reduction

    def _get_estimator(self, seed: int) -> Any:
        if _XGB_AVAILABLE:
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=_get_inner_n_jobs(),
                verbosity=0,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            logger.info("Using GradientBoostingRegressor as XGBoost fallback")
            return GradientBoostingRegressor(random_state=seed)

    def _param_grid(self) -> Dict[str, List[Any]]:
        if self._quick:
            return {
                "model__n_estimators": [100],
                "model__max_depth": [4],
                "model__learning_rate": [0.1],
            }
        grid: Dict[str, List[Any]] = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        }
        if _XGB_AVAILABLE:
            # XGBoost-specific params (not supported by GradientBoostingRegressor)
            grid["model__colsample_bytree"] = [0.8, 1.0]
            grid["model__min_child_weight"] = [1, 5]
        return grid

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        t0 = time.time()
        logger.debug("WF-XGB: train=%d, test=%d, features=%d",
                      len(X_train), len(X_test), X_train.shape[1])

        # Tree models are scale-invariant and do not need StandardScaler.
        # PCA destroys physically meaningful feature axes and hurts
        # interpretability (feature importance becomes meaningless).
        steps: List[Tuple[str, Any]] = [
            ("model", self._get_estimator(seed)),
        ]
        pipe = Pipeline(steps)

        _inner_jobs = _get_inner_n_jobs()
        grid = GridSearchCV(
            pipe,
            self._param_grid(),
            cv=max(2, min(self._n_cv, len(X_train))),
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=_inner_jobs,
            error_score=np.nan,  # skip failing folds instead of raising
        )
        grid.fit(_safe_np(X_train), _safe_np(y_train))

        # ── Early stopping refinement (#3) ──
        # After GridSearchCV finds the best hyperparameters, retrain the
        # best model with early_stopping to find the optimal n_estimators.
        # This reduces overfitting and training cost.
        best_pipe = grid.best_estimator_
        best_model = best_pipe.named_steps["model"]
        used_early_stop = False

        if (
            _XGB_AVAILABLE
            and isinstance(best_model, XGBRegressor)
            and not self._quick
            and len(X_train) >= 20
        ):
            try:
                from sklearn.base import clone as _clone
                from sklearn.model_selection import train_test_split as _tts

                # Apply the scaler (and optionally PCA) from the best pipeline
                # to transform training data for early stopping probe.
                preprocessor = Pipeline(
                    [(name, step) for name, step in best_pipe.steps if name != "model"]
                )
                X_tr_transformed = preprocessor.transform(_safe_np(X_train))

                # Split *training* data into train/validation for early stopping
                # to avoid leaking the held-out test set into model selection.
                X_tr_es, X_val_es, y_tr_es, y_val_es = _tts(
                    X_tr_transformed, _safe_np(y_train),
                    test_size=0.2, random_state=seed,
                )

                # Probe: find optimal n_estimators via early stopping
                es_params = best_model.get_params()
                es_params["n_estimators"] = max(es_params.get("n_estimators", 200), 500)
                es_params["early_stopping_rounds"] = 20
                es_model = XGBRegressor(**es_params)
                es_model.fit(
                    np.ascontiguousarray(X_tr_es),
                    y_tr_es,
                    eval_set=[(np.ascontiguousarray(X_val_es), y_val_es)],
                    verbose=False,
                )
                # Check if early stopping actually triggered (stopped before max budget)
                if hasattr(es_model, "best_iteration") and es_model.best_iteration < es_params["n_estimators"] - 1:
                    # Clone the full pipeline and refit from raw data with
                    # the optimal n_estimators.  This avoids the double-transform
                    # bug that would occur if we inserted a model trained on
                    # pre-transformed data back into the pipeline.
                    # Use a temp variable so best_pipe stays fitted if fit() fails.
                    optimal_n = es_model.best_iteration + 1
                    cloned_pipe = _clone(best_pipe)
                    cloned_pipe.set_params(model__n_estimators=optimal_n)
                    cloned_pipe.fit(_safe_np(X_train), _safe_np(y_train))
                    best_pipe = cloned_pipe  # only reassign after successful fit
                    used_early_stop = True
                    logger.debug(
                        "WF-XGB early stop: best_iteration=%d (was n_estimators=%d)",
                        es_model.best_iteration,
                        es_params.get("n_estimators", 200),
                    )
            except Exception:
                logger.debug("WF-XGB early stopping failed, using GridSearchCV result")

        y_train_pred = best_pipe.predict(_safe_np(X_train))
        y_test_pred = best_pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        # Feature importance from tree model
        model_step = best_pipe.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            fi_raw = model_step.feature_importances_.tolist()
            if "pca" not in best_pipe.named_steps:
                fi = dict(zip(X_train.columns, fi_raw))
            else:
                fi = {f"PC{i}": float(v) for i, v in enumerate(fi_raw)}
        else:
            fi = {}

        result = RunResult(
            workflow=self.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=train_s["rmse"],
            rmse_test=test_s["rmse"],
            mae_train=train_s["mae"],
            mae_test=test_s["mae"],
            r2_train=train_s["r2"],
            r2_test=test_s["r2"],
            y_test_true=_safe_np(y_test).copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params=grid.best_params_,
            artifacts={
                "feature_importance": fi,
                "cv_results_best_score": float(grid.best_score_),
                "early_stopping_used": used_early_stop,
            },
            elapsed_sec=time.time() - t0,
        )
        return result


# ---------------------------------------------------------------------------
# WF-ENS: Seed-varied ensemble
# ---------------------------------------------------------------------------


class WorkflowENS(BaseWorkflow):
    """Seed-varied ensemble workflow for uncertainty quantification.

    Runs the same model template with different random seeds and reports
    mean prediction and prediction standard deviation.

    Purpose:
    - Prediction variance in extrapolation regions
    - Safety evaluation
    """

    name = "WF-ENS"

    def __init__(
        self,
        n_members: int = 5,
        base_workflow: Optional[str] = "xgb",
        quick: bool = False,
        dim_reduction: bool = True,
    ) -> None:
        self._n_members = n_members
        self._base_workflow = base_workflow
        self._quick = quick
        self._dim_reduction = dim_reduction

    def _make_member(self, seed: int, n_features: int = 132) -> Pipeline:
        if self._base_workflow == "xgb" and _XGB_AVAILABLE:
            model = XGBRegressor(
                n_estimators=100 if self._quick else 200,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=_get_inner_n_jobs(),
                verbosity=0,
            )
            # Tree: no scaler / PCA needed
            return Pipeline([("model", model)])
        elif self._base_workflow == "xgb":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100 if self._quick else 200,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed,
            )
            # Tree: no scaler / PCA needed
            return Pipeline([("model", model)])
        else:
            # Linear model (Ridge): keep scaler + optional PCA
            model = Ridge(alpha=1.0)
            steps: List[Tuple[str, Any]] = [
                ("scaler", StandardScaler()),
                *_make_pca_step(n_features, self._dim_reduction),
                ("model", model),
            ]
            return Pipeline(steps)

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        t0 = time.time()
        logger.debug("WF-ENS: train=%d, test=%d, members=%d",
                      len(X_train), len(X_test), self._n_members)

        preds_list: List[np.ndarray] = []
        train_preds_list: List[np.ndarray] = []

        n_features = X_train.shape[1]
        for m in range(self._n_members):
            # Use a large prime multiplier to avoid seed collisions.
            # The old formula (seed * 1000 + m) collides when
            # seed=1001,m=0 and seed=1,m=1000 (both → 1001000).
            member_seed = (seed + m * 10_000_007) % (2**31)
            pipe = self._make_member(member_seed, n_features=n_features)
            pipe.fit(_safe_np(X_train), _safe_np(y_train))
            preds_list.append(pipe.predict(_safe_np(X_test)))
            train_preds_list.append(pipe.predict(_safe_np(X_train)))

        preds_arr = np.stack(preds_list, axis=0)  # (n_members, n_test)
        train_preds_arr = np.stack(train_preds_list, axis=0)

        y_test_pred = preds_arr.mean(axis=0)
        y_test_std = preds_arr.std(axis=0)
        y_train_pred = train_preds_arr.mean(axis=0)

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        result = RunResult(
            workflow=self.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=train_s["rmse"],
            rmse_test=test_s["rmse"],
            mae_train=train_s["mae"],
            mae_test=test_s["mae"],
            r2_train=train_s["r2"],
            r2_test=test_s["r2"],
            y_test_true=_safe_np(y_test).copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params={"n_members": self._n_members, "base": self._base_workflow},
            artifacts={
                "pred_std_test": y_test_std.tolist(),
                "pred_mean_test": y_test_pred.tolist(),
            },
            elapsed_sec=time.time() - t0,
        )
        return result


# ---------------------------------------------------------------------------
# WF-RF: Random Forest
# ---------------------------------------------------------------------------


class WorkflowRF(BaseWorkflow):
    """Random Forest workflow with grid-search hyperparameter optimisation.

    Purpose:
    - Non-linear regression with ensemble of decision trees
    - Feature importance via impurity-based measures
    - Robust to overfitting with proper tuning
    - Captures non-linear interactions without gradient boosting bias

    Uses 10-fold CV by default for hyperparameter tuning.
    """

    name = "WF-RF"

    def __init__(self, n_cv: int = 10, quick: bool = False, dim_reduction: bool = True) -> None:
        self._n_cv = n_cv
        self._quick = quick
        self._dim_reduction = dim_reduction

    def _param_grid(self) -> Dict[str, List[Any]]:
        if self._quick:
            return {
                "model__n_estimators": [100],
                "model__max_depth": [None],
                "model__min_samples_split": [2],
            }
        return {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 15],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 1.0],
        }

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
        **kwargs: Any,
    ) -> RunResult:
        t0 = time.time()
        logger.debug("WF-RF: train=%d, test=%d, features=%d",
                      len(X_train), len(X_test), X_train.shape[1])

        # Tree models are scale-invariant and do not need StandardScaler.
        # PCA destroys physically meaningful feature axes and hurts
        # interpretability (feature importance becomes meaningless).
        _inner_jobs = _get_inner_n_jobs()
        steps: List[Tuple[str, Any]] = [
            ("model", RandomForestRegressor(
                random_state=seed, n_jobs=_inner_jobs,
            )),
        ]
        pipe = Pipeline(steps)

        grid = GridSearchCV(
            pipe,
            self._param_grid(),
            cv=max(2, min(self._n_cv, len(X_train))),
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=_inner_jobs,
            error_score=np.nan,
        )
        grid.fit(_safe_np(X_train), _safe_np(y_train))

        best_pipe = grid.best_estimator_
        y_train_pred = best_pipe.predict(_safe_np(X_train))
        y_test_pred = best_pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        # Feature importance from tree model
        model_step = best_pipe.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            fi_raw = model_step.feature_importances_.tolist()
            if "pca" not in best_pipe.named_steps:
                fi = dict(zip(X_train.columns, fi_raw))
            else:
                fi = {f"PC{i}": float(v) for i, v in enumerate(fi_raw)}
        else:
            fi = {}

        result = RunResult(
            workflow=self.name,
            feature_set=kwargs.get("feature_set", ""),
            split_policy=kwargs.get("split_policy", ""),
            seed=seed,
            fold=kwargs.get("fold", 0),
            rmse_train=train_s["rmse"],
            rmse_test=test_s["rmse"],
            mae_train=train_s["mae"],
            mae_test=test_s["mae"],
            r2_train=train_s["r2"],
            r2_test=test_s["r2"],
            y_test_true=_safe_np(y_test).copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params=grid.best_params_,
            artifacts={
                "feature_importance": fi,
                "cv_results_best_score": float(grid.best_score_),
            },
            elapsed_sec=time.time() - t0,
        )
        return result


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------

WORKFLOW_REGISTRY: Dict[str, type] = {
    "WF-LIN": WorkflowLIN,
    "WF-LASSO": WorkflowLASSO,
    "WF-ARD": WorkflowARD,
    "WF-XGB": WorkflowXGB,
    "WF-RF": WorkflowRF,
    "WF-ENS": WorkflowENS,
}


def get_workflow(name: str, **kwargs: Any) -> BaseWorkflow:
    """Instantiate a workflow by name.

    Parameters
    ----------
    name : str
        One of 'WF-LIN', 'WF-XGB', 'WF-ENS'.
    **kwargs
        Forwarded to the workflow constructor.
    """
    if name not in WORKFLOW_REGISTRY:
        raise ValueError(
            f"Unknown workflow '{name}'. Available: {list(WORKFLOW_REGISTRY.keys())}"
        )
    return WORKFLOW_REGISTRY[name](**kwargs)  # type: ignore[call-arg]
