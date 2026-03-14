"""MLflow experiment tracker — falls back to in-memory store when mlflow is absent."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.info(
        "mlflow not installed — MLflowTracker will use in-memory fallback. "
        "Install with: pip install mlflow"
    )


def is_mlflow_available() -> bool:
    """Return True if the ``mlflow`` package is importable."""
    return _MLFLOW_AVAILABLE


@dataclass
class _InMemoryRun:
    """Single tracked run (in-memory fallback)."""

    run_id: str
    run_name: str
    params: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "RUNNING"


class _InMemoryStore:
    """In-memory replacement for MLflow tracking."""

    def __init__(self) -> None:
        self._runs: Dict[str, _InMemoryRun] = {}
        self._counter: int = 0

    def create_run(self, run_name: str) -> str:
        self._counter += 1
        run_id = f"inmemory-{self._counter:06d}"
        self._runs[run_id] = _InMemoryRun(
            run_id=run_id,
            run_name=run_name,
            start_time=time.time(),
        )
        return run_id

    def log_params(self, run_id: str, params: Dict[str, str]) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.params.update(params)

    def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.metrics.update(metrics)

    def log_artifact(self, run_id: str, path: str) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.artifacts.append(path)

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.tags.update(tags)

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.end_time = time.time()
            run.status = status

    @property
    def runs(self) -> List[_InMemoryRun]:
        return list(self._runs.values())

    def get_run(self, run_id: str) -> Optional[_InMemoryRun]:
        return self._runs.get(run_id)


class MLflowTracker:
    """Unified experiment tracker — delegates to MLflow or in-memory store."""

    def __init__(
        self,
        experiment_name: str = "extrapolation_discovery",
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self._experiment_name = experiment_name
        self._enabled = enabled
        self._use_mlflow = _MLFLOW_AVAILABLE and enabled
        self._current_run_id: Optional[str] = None
        self._fallback = _InMemoryStore()

        if self._use_mlflow:
            if tracking_uri is not None:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(
                "MLflow tracker initialised: experiment=%s, uri=%s",
                experiment_name,
                mlflow.get_tracking_uri(),
            )
        else:
            logger.info(
                "MLflow tracker using in-memory fallback "
                "(experiment=%s)",
                experiment_name,
            )

    @property
    def is_mlflow_active(self) -> bool:
        """Whether real MLflow tracking is being used."""
        return self._use_mlflow

    @property
    def current_run_id(self) -> Optional[str]:
        return self._current_run_id

    # ----- Run lifecycle -----

    def start_run(
        self,
        run_name: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new tracked run.  Returns the run ID."""
        if not self._enabled:
            return "disabled"

        if self._use_mlflow:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._current_run_id = run.info.run_id
        else:
            self._current_run_id = self._fallback.create_run(run_name)
            if tags:
                self._fallback.set_tags(self._current_run_id, tags)

        logger.debug("Started run: %s (%s)", run_name, self._current_run_id)
        return self._current_run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if not self._enabled or self._current_run_id is None:
            return

        if self._use_mlflow:
            mlflow.end_run(status=status)
        else:
            self._fallback.end_run(self._current_run_id, status)

        logger.debug("Ended run: %s (status=%s)", self._current_run_id, status)
        self._current_run_id = None

    @contextmanager
    def run_context(
        self,
        run_name: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[str, None, None]:
        """Context manager for a tracked run. Yields run ID."""
        run_id = self.start_run(run_name=run_name, tags=tags)
        try:
            yield run_id
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")

    # ----- Logging -----

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log run parameters (converted to strings for MLflow)."""
        if not self._enabled or self._current_run_id is None:
            return

        str_params = {k: str(v) for k, v in params.items()}

        if self._use_mlflow:
            mlflow.log_params(str_params)
        else:
            self._fallback.log_params(self._current_run_id, str_params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log run metrics."""
        if not self._enabled or self._current_run_id is None:
            return

        if self._use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            self._fallback.log_metrics(self._current_run_id, metrics)

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None,
    ) -> None:
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Log a local file as a run artifact."""
        if not self._enabled or self._current_run_id is None:
            return

        if self._use_mlflow:
            mlflow.log_artifact(local_path)
        else:
            self._fallback.log_artifact(self._current_run_id, local_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current run."""
        if not self._enabled or self._current_run_id is None:
            return

        if self._use_mlflow:
            mlflow.set_tags(tags)
        else:
            self._fallback.set_tags(self._current_run_id, tags)

    # ----- Convenience: log a RunResult -----

    def log_run_result(self, run_result: Any) -> None:
        """Log a RunResult's params + metrics to the tracker."""
        if not self._enabled:
            return

        self.log_params({
            "workflow": run_result.workflow,
            "feature_set": run_result.feature_set,
            "split_policy": run_result.split_policy,
            "seed": str(run_result.seed),
            "fold": str(run_result.fold),
        })
        self.log_metrics(run_result.metrics_dict())
        self.log_metric("elapsed_sec", run_result.elapsed_sec)

        # Log hyperparameters from the workflow run
        if run_result.params:
            hp = {f"hp_{k}": str(v) for k, v in run_result.params.items()}
            self.log_params(hp)

    # ----- Convenience: log experiment summary -----

    def log_experiment_summary(
        self,
        n_runs: int,
        validity_scores: Sequence[Any],
        ood_results: Dict[str, Any],
        elapsed_sec: float,
    ) -> None:
        """Log experiment-level summary metrics."""
        if not self._enabled:
            return

        self.log_metrics({
            "total_runs": float(n_runs),
            "total_elapsed_sec": elapsed_sec,
        })

        if validity_scores:
            best = validity_scores[0]
            self.log_metrics({
                "best_validity_total": best.total,
                "best_validity_effect_size": best.effect_size,
                "best_validity_stability": best.stability,
                "best_validity_generalisation": best.generalisation,
                "best_validity_extrapolation_safety": best.extrapolation_safety,
            })
            self.set_tags({"best_feature_set": best.feature_set})

        total_ood = sum(r.n_ood for r in ood_results.values())
        total_samples = sum(r.n_total for r in ood_results.values())
        if total_samples > 0:
            self.log_metrics({
                "total_ood_samples": float(total_ood),
                "total_ood_ratio": float(total_ood) / float(total_samples),
            })

    # ----- Query runs (for comparison) -----

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return summary of all tracked runs."""
        if self._use_mlflow:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                return []
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
            )
            return [
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name or "",
                    "params": dict(r.data.params),
                    "metrics": dict(r.data.metrics),
                    "status": r.info.status,
                }
                for r in runs
            ]
        else:
            return [
                {
                    "run_id": r.run_id,
                    "run_name": r.run_name,
                    "params": dict(r.params),
                    "metrics": dict(r.metrics),
                    "status": r.status,
                }
                for r in self._fallback.runs
            ]

    def get_tracking_uri(self) -> str:
        """Return the tracking URI (or 'in-memory' for fallback)."""
        if self._use_mlflow:
            return str(mlflow.get_tracking_uri())
        return "in-memory"
