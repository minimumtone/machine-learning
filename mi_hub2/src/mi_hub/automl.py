"""mi_hub.automl — FLAML ベースの AutoML ラッパー(Phase 4)。

方針:
  - まず FLAML(依存が軽い)。AutoGluon は別 conda 環境で必要時のみ。
  - 学習は必ず tracking.track() の中で行い、best config / CV 指標 /
    ホールドアウト指標 / モデル本体 を MLflow に残す。
  - 特徴量は datastore の kind (例 "hea_features") か Feast の
    get_historical_features() 出力をそのまま渡せる。

使用例:
    from mi_hub import automl, datastore as ds
    df = ds.load("hea_features")
    res = automl.fit_baseline(df, target="yield_strength",
                              task="regression", time_budget=120,
                              drop=ds.PROVENANCE_COLS)
    print(res["holdout_metrics"], res["best_estimator"])
"""
from __future__ import annotations

import pandas as pd

from . import datastore as ds
from . import tracking as tr


def fit_baseline(df: pd.DataFrame, *, target: str, task: str = "regression",
                 time_budget: int = 60, test_size: float = 0.2,
                 drop: list[str] | None = None, experiment: str = "automl",
                 seed: int = 0) -> dict:
    try:
        from flaml import AutoML
    except ImportError as e:
        raise ImportError("pip install 'flaml[automl]' が必要です。") from e
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
    from sklearn.model_selection import train_test_split

    drop = [c for c in (drop or []) if c in df.columns]
    X = df.drop(columns=drop + [target]).select_dtypes("number")
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    rid = ds.new_run_id()
    with tr.track(experiment, run_id=rid,
                  params={"target": target, "task": task,
                          "time_budget": time_budget, "n_features": X.shape[1],
                          "n_train": len(X_tr)}):
        # FLAML 自身の MLflow 自動ロギングは skops 直列化で失敗しやすく、
        # 記録は mi_hub 側で一元化するため無効にする。
        am = AutoML(mlflow_logging=False)
        am.fit(X_tr, y_tr, task=task, time_budget=time_budget, seed=seed,
               verbose=0, mlflow_logging=False)
        pred = am.predict(X_te)
        if task == "regression":
            metrics = {"mae": mean_absolute_error(y_te, pred),
                       "r2": r2_score(y_te, pred)}
        else:
            metrics = {"accuracy": accuracy_score(y_te, pred)}
        tr.log_metrics(metrics)
        import mlflow
        mlflow.log_params({"best_estimator": am.best_estimator})
        mlflow.sklearn.log_model(am.model, name="model",
                                 serialization_format="cloudpickle")
        pred_df = X_te.copy()
        pred_df[target] = y_te.values
        pred_df["prediction"] = pred
        ds.save(pred_df, "predictions", run_id=rid, source="flaml")
        tr.log_table(pred_df, "holdout_predictions.parquet")

    return {"run_id": rid, "best_estimator": am.best_estimator,
            "best_config": am.best_config, "holdout_metrics": metrics}
