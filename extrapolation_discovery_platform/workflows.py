"""ML Workflow Module for Extrapolation Discovery Platform.

Workflow templates:
  WF-LIN   - Linear regression (Ridge, coefficient analysis)
  WF-LASSO - Lasso regression (L1 sparse feature selection)
  WF-ARD   - Bayesian ARD regression (automatic relevance determination)
  WF-XGB   - XGBoost with hyperparameter optimisation
  WF-ENS   - Seed-varied ensemble for uncertainty quantification
  WF-RF    - Random Forest with hyperparameter optimisation
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
from sklearn.linear_model import ARDRegression, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


from extrapolation_discovery_platform._utils import (  # noqa: E402
    get_safe_n_jobs,
    safe_array as _safe_np,
)


# ---------------------------------------------------------------------------
# XGBoost / fallback factory  (Issue M — single creation point)
# ---------------------------------------------------------------------------

try:
    from xgboost import XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed -- WF-XGB will fall back to GradientBoosting")


def _make_xgb_or_fallback(seed: int, n_jobs: int = 1, **kwargs) -> object:
    """Create an XGBRegressor or GradientBoostingRegressor fallback."""
    if _XGB_AVAILABLE:
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
            **kwargs,
        )
    from sklearn.ensemble import GradientBoostingRegressor
    _GB_PARAMS = {"n_estimators", "max_depth", "learning_rate", "subsample", "random_state"}
    gb_kwargs = {k: v for k, v in kwargs.items() if k in _GB_PARAMS}
    return GradientBoostingRegressor(random_state=seed, **gb_kwargs)


# ---------------------------------------------------------------------------
# Run result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Container for a single workflow run result."""

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
    """Compute RMSE, MAE, R2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Shared helpers  (Issues A, B, C)
# ---------------------------------------------------------------------------


def _safe_std_y(y_train: np.ndarray) -> float:
    """Compute sample std of y, clamped to 1.0 for degenerate cases."""
    if len(y_train) > 1:
        std_y = float(np.std(y_train, ddof=1))
    else:
        std_y = 0.0
    return std_y if std_y >= 1e-12 else 1.0


def _coef_to_dict(
    columns: pd.Index,
    coef: np.ndarray,
    pipe: Pipeline,
) -> Dict[str, float]:
    """Map coefficient array to a {name: value} dict, handling PCA."""
    if "pca" not in pipe.named_steps:
        return dict(zip(columns, coef.tolist()))
    return {f"PC{i}": float(c) for i, c in enumerate(coef)}


def _make_result(
    workflow_name: str,
    train_s: Dict[str, float],
    test_s: Dict[str, float],
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    t0: float,
    seed: int,
    params: Dict[str, Any],
    artifacts: Dict[str, Any],
    **kwargs: Any,
) -> RunResult:
    """Build a RunResult from common arguments (eliminates 6x duplication)."""
    return RunResult(
        workflow=workflow_name,
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
        params=params,
        artifacts=artifacts,
        elapsed_sec=time.time() - t0,
    )


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
# Dimensionality-reduction helper
# ---------------------------------------------------------------------------


def _make_pca_step(
    n_features: int,
    dim_reduction: bool,
    variance_ratio: float = 0.95,
) -> List[Tuple[str, Any]]:
    """Return pipeline steps for optional PCA dimensionality reduction."""
    if not dim_reduction or n_features <= 2:
        return []
    return [("pca", PCA(n_components=variance_ratio, svd_solver="full"))]


# ---------------------------------------------------------------------------
# WF-LIN: Linear regression
# ---------------------------------------------------------------------------


class WorkflowLIN(BaseWorkflow):
    """Linear regression workflow (Ridge).

    ``alpha=None``（既定）の場合は RidgeCV で正則化強度をデータから選択する。
    固定 alpha=1.0 は外挿条件では正則化が弱すぎ、予測が訓練範囲を大きく
    超える原因になるため。
    """

    name = "WF-LIN"

    def __init__(self, alpha: Optional[float] = None, dim_reduction: bool = True) -> None:
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

        if self._alpha is None:
            model_step: Any = RidgeCV(alphas=np.logspace(-2, 4, 25))
        else:
            model_step = Ridge(alpha=self._alpha)
        steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            *_make_pca_step(X_train.shape[1], self._dim_reduction),
            ("model", model_step),
        ]
        pipe = Pipeline(steps)
        pipe.fit(_safe_np(X_train), _safe_np(y_train))

        y_train_pred = pipe.predict(_safe_np(X_train))
        y_test_pred = pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        model = pipe.named_steps["model"]
        coef_raw = model.coef_
        effective_alpha = (
            float(model.alpha_) if self._alpha is None else float(self._alpha)
        )
        std_y = _safe_std_y(_safe_np(y_train))
        coef_std = coef_raw / std_y

        return _make_result(
            self.name, train_s, test_s,
            y_test, y_test_pred, t0, seed,
            params={"alpha": effective_alpha, "dim_reduction": self._dim_reduction},
            artifacts={
                "coef_raw": _coef_to_dict(X_train.columns, coef_raw, pipe),
                "coef_std": _coef_to_dict(X_train.columns, coef_std, pipe),
                "residuals_test": (_safe_np(y_test) - y_test_pred).tolist(),
                "n_components": (
                    pipe.named_steps["pca"].n_components_
                    if "pca" in pipe.named_steps else X_train.shape[1]
                ),
            },
            **kwargs,
        )


# ---------------------------------------------------------------------------
# WF-LASSO: Lasso regression (L1 regularisation)
# ---------------------------------------------------------------------------


class WorkflowLASSO(BaseWorkflow):
    """Lasso (L1) regression workflow with LassoCV for alpha selection."""

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
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            *_make_pca_step(X_train.shape[1], self._dim_reduction),
            ("model", LassoCV(
                cv=max(2, min(10, len(X_train))),
                random_state=seed, max_iter=10000,
            )),
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
        std_y = _safe_std_y(_safe_np(y_train))
        coef_std = coef_raw / std_y

        return _make_result(
            self.name, train_s, test_s,
            y_test, y_test_pred, t0, seed,
            params={"alpha": float(model.alpha_), "dim_reduction": self._dim_reduction},
            artifacts={
                "coef_raw": _coef_to_dict(X_train.columns, coef_raw, pipe),
                "coef_std": _coef_to_dict(X_train.columns, coef_std, pipe),
                "n_nonzero_features": n_nonzero,
                "residuals_test": (_safe_np(y_test) - y_test_pred).tolist(),
                "n_components": (
                    pipe.named_steps["pca"].n_components_
                    if "pca" in pipe.named_steps else X_train.shape[1]
                ),
            },
            **kwargs,
        )


# ---------------------------------------------------------------------------
# WF-ARD: Automatic Relevance Determination (Bayesian sparse regression)
# ---------------------------------------------------------------------------


class WorkflowARD(BaseWorkflow):
    """ARD Bayesian regression workflow."""

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
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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
        relevance = 1.0 / (model.lambda_ + 1e-10)
        relevance_norm = relevance / relevance.max() if relevance.max() > 0 else relevance
        std_y = _safe_std_y(_safe_np(y_train))
        coef_std = coef_raw / std_y

        return _make_result(
            self.name, train_s, test_s,
            y_test, y_test_pred, t0, seed,
            params={"dim_reduction": self._dim_reduction},
            artifacts={
                "coef_raw": _coef_to_dict(X_train.columns, coef_raw, pipe),
                "coef_std": _coef_to_dict(X_train.columns, coef_std, pipe),
                "relevance_scores": _coef_to_dict(X_train.columns, relevance_norm, pipe),
                "residuals_test": (_safe_np(y_test) - y_test_pred).tolist(),
                "n_components": (
                    pipe.named_steps["pca"].n_components_
                    if "pca" in pipe.named_steps else X_train.shape[1]
                ),
            },
            **kwargs,
        )


# ---------------------------------------------------------------------------
# WF-XGB: XGBoost + HPO
# ---------------------------------------------------------------------------


class WorkflowXGB(BaseWorkflow):
    """XGBoost workflow with grid-search hyperparameter optimisation."""

    name = "WF-XGB"

    def __init__(self, n_cv: int = 10, quick: bool = False, dim_reduction: bool = True) -> None:
        self._n_cv = n_cv
        self._quick = quick
        self._dim_reduction = dim_reduction

    def _get_estimator(self, seed: int) -> object:
        return _make_xgb_or_fallback(seed, n_jobs=get_safe_n_jobs())

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

        # Bug#3 fix: add StandardScaler before XGB.
        # XGB is scale-invariant for tree splits, but MAGPIE features span
        # many orders of magnitude.  Without scaling, features with large
        # raw values dominate the gain calculation and suppress informative
        # features — causing WF-XGB to produce the same predictions as
        # simpler models that happened to be seeded with the same data.
        steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", self._get_estimator(seed)),
        ]
        pipe = Pipeline(steps)

        _inner_jobs = get_safe_n_jobs()
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

                pre_steps = [
                    (name, step) for name, step in best_pipe.steps
                    if name != "model"
                ]
                if pre_steps:
                    preprocessor = Pipeline(pre_steps)
                    X_tr_transformed = preprocessor.transform(_safe_np(X_train))
                else:
                    X_tr_transformed = _safe_np(X_train)

                X_tr_es, X_val_es, y_tr_es, y_val_es = _tts(
                    X_tr_transformed, _safe_np(y_train),
                    test_size=0.2, random_state=seed,
                )

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
                if (
                    hasattr(es_model, "best_iteration")
                    and es_model.best_iteration < es_params["n_estimators"] - 1
                ):
                    optimal_n = es_model.best_iteration + 1
                    cloned_pipe = _clone(best_pipe)
                    cloned_pipe.set_params(model__n_estimators=optimal_n)
                    cloned_pipe.fit(_safe_np(X_train), _safe_np(y_train))
                    best_pipe = cloned_pipe
                    used_early_stop = True
                    logger.debug(
                        "WF-XGB early stop: best_iteration=%d",
                        es_model.best_iteration,
                    )
            except Exception:
                logger.debug("WF-XGB early stopping failed, using GridSearchCV result")

        y_train_pred = best_pipe.predict(_safe_np(X_train))
        y_test_pred = best_pipe.predict(_safe_np(X_test))

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        model_step = best_pipe.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            fi = _coef_to_dict(
                X_train.columns, model_step.feature_importances_, best_pipe,
            )
        else:
            fi = {}

        return _make_result(
            self.name, train_s, test_s,
            y_test, y_test_pred, t0, seed,
            params=grid.best_params_,
            artifacts={
                "feature_importance": fi,
                "cv_results_best_score": float(grid.best_score_),
                "early_stopping_used": used_early_stop,
            },
            **kwargs,
        )


# ---------------------------------------------------------------------------
# WF-ENS: Seed-varied ensemble
# ---------------------------------------------------------------------------


class WorkflowENS(BaseWorkflow):
    """Seed-varied GradientBoosting ensemble for uncertainty quantification.

    Each member uses a GradientBoostingRegressor with a different random_state.
    GBR uses stochastic subsampling (subsample < 1.0), so different seeds
    produce genuinely different predictions — unlike Ridge which is fully
    deterministic and would make all members identical.

    Why GradientBoosting instead of Ridge or XGB:
      - Ridge is deterministic → all members predict identically → ENS == LIN
      - XGB (same params as WF-XGB) → ENS == XGB
      - GradientBoostingRegressor with subsample=0.8:
          * non-deterministic across seeds (stochastic gradient boosting)
          * distinct from both linear (LIN/LASSO/ARD) and XGB results
          * fast with quick=True (n_estimators=50, max_depth=3)
          * still provides meaningful pred uncertainty via inter-member std
    """

    name = "WF-ENS"

    def __init__(
        self,
        n_members: int = 5,
        base_workflow: Optional[str] = "gbr",   # GradientBoosting — truly stochastic
        quick: bool = False,
        dim_reduction: bool = True,
    ) -> None:
        self._n_members = n_members
        self._base_workflow = base_workflow
        self._quick = quick
        self._dim_reduction = dim_reduction

    def _make_member(self, seed: int, n_features: int = 132) -> Pipeline:
        from sklearn.ensemble import GradientBoostingRegressor
        if self._base_workflow == "xgb":
            model = _make_xgb_or_fallback(
                seed,
                n_jobs=get_safe_n_jobs(),
                n_estimators=100 if self._quick else 200,
                max_depth=4,
                learning_rate=0.1,
            )
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", model),
            ])
        else:
            # GradientBoostingRegressor: subsample=0.8 → stochastic → seed matters
            n_est = 50 if self._quick else 200
            model = GradientBoostingRegressor(
                n_estimators=n_est,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,          # stochastic subsampling: seed changes predictions
                random_state=seed,
            )
            steps: List[Tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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
            member_seed = (seed + m * 10_000_007) % (2**31)
            pipe = self._make_member(member_seed, n_features=n_features)
            pipe.fit(_safe_np(X_train), _safe_np(y_train))
            preds_list.append(pipe.predict(_safe_np(X_test)))
            train_preds_list.append(pipe.predict(_safe_np(X_train)))

        preds_arr = np.stack(preds_list, axis=0)
        train_preds_arr = np.stack(train_preds_list, axis=0)

        y_test_pred = preds_arr.mean(axis=0)
        y_test_std = preds_arr.std(axis=0)
        y_train_pred = train_preds_arr.mean(axis=0)

        train_s = _score(_safe_np(y_train), y_train_pred)
        test_s = _score(_safe_np(y_test), y_test_pred)

        return _make_result(
            self.name, train_s, test_s,
            y_test, y_test_pred, t0, seed,
            params={"n_members": self._n_members, "base": self._base_workflow},
            artifacts={
                "pred_std_test": y_test_std.tolist(),
                "pred_mean_test": y_test_pred.tolist(),
            },
            **kwargs,
        )


# ---------------------------------------------------------------------------
# WF-RF: Random Forest
# ---------------------------------------------------------------------------


class WorkflowRF(BaseWorkflow):
    """Random Forest workflow with grid-search HPO."""

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

        _inner_jobs = get_safe_n_jobs()
        # Bug#3 fix: add StandardScaler before RandomForest.
        # RF is scale-invariant for Gini/variance splits but adding a scaler
        # makes the pipeline consistent with WF-LIN/LASSO/ARD and prevents
        # MAGPIE features with disparate magnitudes from skewing the
        # max_features sampling step (which uses raw feature indices).
        steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
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

        model_step = best_pipe.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            fi = _coef_to_dict(
                X_train.columns, model_step.feature_importances_, best_pipe,
            )
        else:
            fi = {}

        return _make_result(
            self.name, train_s, test_s,
            y_test, y_test_pred, t0, seed,
            params=grid.best_params_,
            artifacts={
                "feature_importance": fi,
                "cv_results_best_score": float(grid.best_score_),
            },
            **kwargs,
        )


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
    """Instantiate a workflow by name."""
    if name not in WORKFLOW_REGISTRY:
        raise ValueError(
            f"Unknown workflow '{name}'. Available: {list(WORKFLOW_REGISTRY.keys())}"
        )
    return WORKFLOW_REGISTRY[name](**kwargs)  # type: ignore[call-arg]
