# %% [markdown]
# # EDA(pygwalker) + AutoML(FLAML) デモ
# datastore に溜めた特徴量テーブルを可視化し、ベースラインモデルを自動構築、
# 結果は MLflow に自動記録される。
# runcell への依頼例: 「hea_features を読み込んで yield_strength の
# ベースラインを FLAML で作り、MLflow の run URL を教えて」

# %% 合成 HEA 特徴量(実運用では MAGPIE/Ωsf テーブルを ds.save しておく)
import numpy as np
import pandas as pd
from mi_hub import datastore as ds

rng = np.random.default_rng(1)
n = 300
feat = pd.DataFrame({
    "alloy_id": [f"HEA{i:04d}" for i in range(n)],
    "vec": rng.uniform(4.5, 8.5, n),           # 価電子濃度
    "delta_r": rng.uniform(0.01, 0.08, n),     # 原子半径ミスマッチ
    "dH_mix": rng.uniform(-20, 5, n),
    "omega_sf_mean": rng.uniform(-5, 15, n),   # Ωsf 平均(ダミー)
})
feat["yield_strength"] = (
    900 - 60 * feat["vec"] + 4000 * feat["delta_r"]
    - 8 * feat["dH_mix"] + 6 * feat["omega_sf_mean"]
    + rng.normal(0, 40, n))
ds.save(feat, "hea_features", source="manual", code_ver="demo")

# %% pygwalker(Phase 1 環境で)
# import pygwalker as pyg
# pyg.walk(ds.load("hea_features"))

# %% FLAML ベースライン(pip install 'flaml[automl]' 後)
from mi_hub import automl
res = automl.fit_baseline(
    ds.load("hea_features"), target="yield_strength",
    task="regression", time_budget=30,
    drop=ds.PROVENANCE_COLS + ["alloy_id"])
print(res["best_estimator"], res["holdout_metrics"])

# %% MLflow の run 一覧を DataFrame で(pygwalker に渡せる)
from mi_hub import tracking as tr
print(tr.runs("automl")[["run_id", "metrics.mae", "metrics.r2",
                         "params.best_estimator"]].head())
