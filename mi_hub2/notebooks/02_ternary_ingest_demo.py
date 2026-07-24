# %% [markdown]
# # TC-Python 三元系等温断面 → mi_hub 流し込みデモ
# 既存の三元断面パイプラインの出力(組成グリッド + 相分類)を
# datastore + MLflow に記録する最小例。
# tc_python が無い環境でも動くよう、合成データのドライランを既定とする。
# 実運用では `compute_section()` を実パイプライン呼び出しに差し替えるだけ。

# %%
import numpy as np
import pandas as pd
from mi_hub import datastore as ds, tracking as tr

SYSTEM = ("Al", "Co", "Cr")
T_K = 1273
DATABASE = "TCHEA7"


def compute_section(system, T, database) -> pd.DataFrame:
    """実運用ではここを TC-Python 呼び出しに置換。
    返り値の規約: x_<el1>, x_<el2>, x_<el3>, phase_label, n_phases
    """
    rng = np.random.default_rng(0)
    n = 500
    a = rng.dirichlet(np.ones(3), size=n)
    labels = np.where(a[:, 0] > 0.5, "BCC_B2",
              np.where(a[:, 1] > 0.5, "FCC_L12", "FCC_A1+BCC_A2"))
    return pd.DataFrame({
        f"x_{system[0]}": a[:, 0],
        f"x_{system[1]}": a[:, 1],
        f"x_{system[2]}": a[:, 2],
        "phase_label": labels,
        "n_phases": np.char.count(labels.astype(str), "+") + 1,
    })


# %% 計算 → 記録(この 8 行が全パイプライン共通の定型)
rid = ds.new_run_id()
with tr.track("ternary_sections", run_id=rid,
              params={"system": "-".join(SYSTEM), "T_K": T_K,
                      "database": DATABASE}):
    df = compute_section(SYSTEM, T_K, DATABASE)
    ds.save(df, "ternary_sections", run_id=rid, source="tc_python",
            code_ver="demo")
    tr.log_table(df, "section.parquet")
    tr.log_metrics({"n_points": len(df),
                    "n_unique_phases": df["phase_label"].nunique()})
print("run_id:", rid)

# %% 在庫確認
print(ds.catalog())

# %% pygwalker で探索(Phase 1 環境で実行)
# import pygwalker as pyg
# pyg.walk(ds.load("ternary_sections"))
