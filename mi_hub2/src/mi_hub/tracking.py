"""mi_hub.tracking — MLflow 記録層。

方針:
  - tracking URI は env MI_HUB_MLFLOW (既定 sqlite:///<data_root>/mlflow.db)。
  - datastore.save() で書いた parquet をそのまま artifact 化し、
    run_id を MLflow の tag "mi_hub.run_id" に載せて相互参照する。
  - TC-Python / OptiMat / AutoML のどの出力も同じ流儀で記録する。

使用例:
    from mi_hub import datastore as ds, tracking as tr

    rid = ds.new_run_id()
    with tr.track("ternary_sections", run_id=rid,
                  params={"database": "TCHEA7", "T": 1273}) as run:
        df = compute_something()
        path = ds.save(df, "ternary_sections", run_id=rid, source="tc_python")
        tr.log_table(df, "section.parquet")
        tr.log_metrics({"n_points": len(df)})
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd

from . import datastore as ds


def tracking_uri() -> str:
    return os.environ.get("MI_HUB_MLFLOW", f"sqlite:///{ds.data_root() / 'mlflow.db'}")


def _setup(experiment: str) -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)


@contextmanager
def track(experiment: str, *, run_id: str | None = None,
          params: dict | None = None, tags: dict | None = None):
    """MLflow run を開き、mi_hub.run_id タグで datastore と紐づける。"""
    _setup(experiment)
    run_id = run_id or ds.new_run_id()
    all_tags = {"mi_hub.run_id": run_id, **(tags or {})}
    with mlflow.start_run(tags=all_tags) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_table(df: pd.DataFrame, name: str = "table.parquet") -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / name
        df.to_parquet(p, index=False)
        mlflow.log_artifact(str(p))


def log_metrics(metrics: dict) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def log_file(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def runs(experiment: str) -> pd.DataFrame:
    """実験の全 run を DataFrame で返す(pygwalker でそのまま可視化可)。"""
    _setup(experiment)
    return mlflow.search_runs(search_all_experiments=False)
