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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

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
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


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


class WorkflowLIN(BaseWorkflow):
    """Linear regression workflow.

    Purpose:
    - Feature sign validation
    - Monotonicity check
    - Leak detection via residual analysis

    Uses Ridge (alpha=1.0) for numerical stability.
    """

    name = "WF-LIN"

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha

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

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=self._alpha)),
        ])
        pipe.fit(X_train.values, y_train.values)

        y_train_pred = pipe.predict(X_train.values)
        y_test_pred = pipe.predict(X_test.values)

        train_s = _score(y_train.values, y_train_pred)
        test_s = _score(y_test.values, y_test_pred)

        # Extract standardised coefficients
        scaler: StandardScaler = pipe.named_steps["scaler"]
        model: Ridge = pipe.named_steps["model"]
        coef_raw = model.coef_
        std_X = scaler.scale_
        std_y = float(np.std(y_train.values)) if np.std(y_train.values) > 0 else 1.0
        coef_std = coef_raw * std_X / std_y

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
            y_test_true=y_test.values.copy(),
            y_test_pred=y_test_pred.copy(),
            test_indices=kwargs.get("test_indices"),
            params={"alpha": self._alpha},
            artifacts={
                "coef_raw": dict(zip(X_train.columns, coef_raw.tolist())),
                "coef_std": dict(zip(X_train.columns, coef_std.tolist())),
                "residuals_test": (y_test.values - y_test_pred).tolist(),
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

    def __init__(self, n_cv: int = 3, quick: bool = False) -> None:
        self._n_cv = n_cv
        self._quick = quick

    def _get_estimator(self, seed: int) -> Any:
        if _XGB_AVAILABLE:
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=1,
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
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
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
        logger.debug("WF-XGB: train=%d, test=%d, features=%d",
                      len(X_train), len(X_test), X_train.shape[1])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", self._get_estimator(seed)),
        ])

        grid = GridSearchCV(
            pipe,
            self._param_grid(),
            cv=min(self._n_cv, len(X_train)),
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        grid.fit(X_train.values, y_train.values)

        best_pipe = grid.best_estimator_
        y_train_pred = best_pipe.predict(X_train.values)
        y_test_pred = best_pipe.predict(X_test.values)

        train_s = _score(y_train.values, y_train_pred)
        test_s = _score(y_test.values, y_test_pred)

        # Feature importance from tree model
        model_step = best_pipe.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            fi = dict(zip(
                X_train.columns,
                model_step.feature_importances_.tolist(),
            ))
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
            y_test_true=y_test.values.copy(),
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
    ) -> None:
        self._n_members = n_members
        self._base_workflow = base_workflow
        self._quick = quick

    def _make_member(self, seed: int) -> Pipeline:
        if self._base_workflow == "xgb" and _XGB_AVAILABLE:
            model = XGBRegressor(
                n_estimators=100 if self._quick else 200,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
        elif self._base_workflow == "xgb":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100 if self._quick else 200,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed,
            )
        else:
            # Ridge has no randomness; random_state is not a valid parameter.
            model = Ridge(alpha=1.0)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

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

        for m in range(self._n_members):
            member_seed = seed * 1000 + m
            pipe = self._make_member(member_seed)
            pipe.fit(X_train.values, y_train.values)
            preds_list.append(pipe.predict(X_test.values))
            train_preds_list.append(pipe.predict(X_train.values))

        preds_arr = np.stack(preds_list, axis=0)  # (n_members, n_test)
        train_preds_arr = np.stack(train_preds_list, axis=0)

        y_test_pred = preds_arr.mean(axis=0)
        y_test_std = preds_arr.std(axis=0)
        y_train_pred = train_preds_arr.mean(axis=0)

        train_s = _score(y_train.values, y_train_pred)
        test_s = _score(y_test.values, y_test_pred)

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
            y_test_true=y_test.values.copy(),
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
# Workflow registry
# ---------------------------------------------------------------------------

WORKFLOW_REGISTRY: Dict[str, type] = {
    "WF-LIN": WorkflowLIN,
    "WF-XGB": WorkflowXGB,
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
    return WORKFLOW_REGISTRY[name](**kwargs)
