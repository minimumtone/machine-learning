#!/usr/bin/env python3
"""Alonso式の補正項を「二元系を厳密に再現する規格化」で読み替える。

Alonso式（式10）の補正項に二元系の Omega_sf をそのまま代入したとき、
その式は元の二元系の体積を再現しない。二元系極限では

  model excess  = c_A c_B (V_A + V_B) Omega
  true  excess  = Omega * (c_A V_A + c_B V_B)

であり、比 kappa = c_A c_B (V_A + V_B) / (c_A V_A + c_B V_B) は
Omega に依存しない純粋な組成・体積の幾何因子である。
50:50参照（BCC SQS、B2）では kappa = 1/2 が厳密に成立するので、
二元系を厳密に再現する校正定数は q_exact = 1/kappa = 2 である。
L1_2 参照（3:1）では kappa は端点体積に依存し、q_exact ~ 8/3 となる。

したがって、校正で得られる q をそのまま「1に近いか」で論じるのではなく、
移行率 f = q / q_exact
（f = 1 は「二元系の超過体積が対ごとに加法的にHEAへ移る」ことを意味する）
で構造間・参照間を比較できる。

出力: paper/pairwise_additivity_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))

from hea_lattice_xgboost import (  # noqa: E402
    ALONSO_TABLE2, INDEPENDENT_TEST, KING_ATOMIC_VOLUMES,
    compute_eq10_scaled,
)
from generate_all_figures import (  # noqa: E402
    load_compounds, compute_omega_sf_pairwise, load_sqs_data,
)


def kappa(c_a: float, c_b: float, v_a: float, v_b: float) -> float:
    """model excess / true excess（Omega に依存しない幾何因子）。"""
    return c_a * c_b * (v_a + v_b) / (c_a * v_a + c_b * v_b)


def rmse(heas, omega, struct, q) -> float:
    sel = [h for h in heas if h["struct"] == struct]
    y = np.array([h["a_exp"] for h in sel])
    p = np.array([compute_eq10_scaled(h["comp"], struct, omega, q)
                  for h in sel])
    return float(np.sqrt(np.mean((p - y) ** 2)))


def optimize_q(heas, omega, struct) -> float:
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda q: rmse(heas, omega, struct, q),
                          bounds=(-10, 10), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.x)


def verify_binary_reproduction(sqs) -> dict:
    """BCC SQS 50:50の実データで q_exact = 2 を数値検証する。"""
    omega = sqs["omega_king"]
    out = {}
    for q in (1.0, 2.0):
        errs = []
        for pair, a_sqs in sqs["sqs_a"].items():
            if pair not in omega:
                continue
            v_true = a_sqs ** 3 / 2
            a_pred = compute_eq10_scaled({pair[0]: 0.5, pair[1]: 0.5},
                                         "BCC", omega, q)
            errs.append(abs(a_pred ** 3 / 2 - v_true) / v_true * 100)
        errs = np.array(errs)
        out[f"q{q:.0f}"] = {
            "n_pairs": int(errs.size),
            "median_abs_vol_error_pct": float(np.median(errs)),
            "max_abs_vol_error_pct": float(np.max(errs)),
        }
    return out


def l12_kappa_stats(all_df) -> dict:
    ks, counts = [], {}
    sub = all_df[all_df["stype"] == "L12"]
    for _, row in sub.iterrows():
        el_a, el_b = row.get("element_A", ""), row.get("element_B", "")
        if el_a not in KING_ATOMIC_VOLUMES or el_b not in KING_ATOMIC_VOLUMES:
            continue
        if el_a == el_b:
            continue
        c_a, c_b = float(row.get("count_A", 3)), float(row.get("count_B", 1))
        total = c_a + c_b
        counts[f"{int(c_a)}:{int(c_b)}"] = counts.get(
            f"{int(c_a)}:{int(c_b)}", 0) + 1
        ks.append(kappa(c_a / total, c_b / total,
                        KING_ATOMIC_VOLUMES[el_a], KING_ATOMIC_VOLUMES[el_b]))
    ks = np.array(ks)
    return {
        "n_rows": int(ks.size),
        "count_ratio_histogram": counts,
        "kappa_median": float(np.median(ks)),
        "kappa_min": float(np.min(ks)),
        "kappa_max": float(np.max(ks)),
        "q_exact_median": float(1.0 / np.median(ks)),
        "q_exact_min": float(1.0 / np.max(ks)),
        "q_exact_max": float(1.0 / np.min(ks)),
    }


def main() -> None:
    all_df = load_compounds()
    ob2, ol12 = compute_omega_sf_pairwise(all_df)
    sqs = load_sqs_data()

    models = {
        "BCC_SQS_DFT_vegard": {
            "struct": "BCC", "omega": sqs["omega_dft"],
            "reference_composition": "50:50", "q_exact": 2.0,
            "q_adopted": 1.0,
        },
        "BCC_SQS_King": {
            "struct": "BCC", "omega": sqs["omega_king"],
            "reference_composition": "50:50", "q_exact": 2.0,
            "q_adopted": None,
        },
        "BCC_B2_King": {
            "struct": "BCC", "omega": ob2,
            "reference_composition": "50:50", "q_exact": 2.0,
            "q_adopted": None,
        },
        "FCC_SQS_DFT_vegard": {
            "struct": "FCC", "omega": sqs["fcc_omega_dft"],
            "reference_composition": "50:50", "q_exact": 2.0,
            "q_adopted": None,
        },
        "FCC_L12_King": {
            "struct": "FCC", "omega": ol12,
            "reference_composition": "3:1", "q_exact": None,
            "q_adopted": None,
        },
    }
    l12 = l12_kappa_stats(all_df)
    models["FCC_L12_King"]["q_exact"] = l12["q_exact_median"]

    out = {
        "l12_kappa": l12,
        "binary_reproduction_check_bcc_sqs": verify_binary_reproduction(sqs),
        "models": {},
    }
    for name, spec in models.items():
        struct, omega = spec["struct"], spec["omega"]
        q_opt = optimize_q(ALONSO_TABLE2, omega, struct)
        q_exact = spec["q_exact"]
        entry = {
            "structure": struct,
            "reference_composition": spec["reference_composition"],
            "n_pairs": len(omega),
            "q_exact": q_exact,
            "q_opt_calibration": q_opt,
            "transfer_fraction_f": q_opt / q_exact,
            "rmse_calibration_qopt_A": rmse(ALONSO_TABLE2, omega, struct,
                                            q_opt),
            "rmse_test_qopt_A": rmse(INDEPENDENT_TEST, omega, struct, q_opt),
            "rmse_test_q_exact_A": rmse(INDEPENDENT_TEST, omega, struct,
                                        q_exact),
            "rmse_test_vegard_A": rmse(INDEPENDENT_TEST, omega, struct, 0.0),
        }
        if spec["q_adopted"] is not None:
            entry["q_adopted"] = spec["q_adopted"]
            entry["transfer_fraction_f_adopted"] = (
                spec["q_adopted"] / q_exact
            )
            entry["rmse_test_q_adopted_A"] = rmse(
                INDEPENDENT_TEST, omega, struct, spec["q_adopted"]
            )
        out["models"][name] = entry

    with (PAPER / "pairwise_additivity_metrics.json").open("w") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
