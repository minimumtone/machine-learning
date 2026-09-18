#!/usr/bin/env python3
"""Redlich--Kister一次項でAlonso式を拡張し、HEA格子定数予測を再評価する。

Alonso式（式10）の二元系極限は超過体積が x(1-x) に比例する対称形であり、
組成非対称性を表現できない。SQSの3組成データ（x=0.25/0.50/0.75）から
非対称係数 a1 を測定し、次の拡張形を独立テストで評価する。

  V = n_auc [ Sum_i c_i V_i
              + q  Sum_{i != j} c_i c_j V_j Omega_ij
              + q1 Sum_{i<j} c_i c_j (c_i - c_j) (V_i + V_j) omega1_ij ]

omega1_ij = a1_ij / (V_A^SQS + V_B^SQS)  （無次元化。i はアルファベット順先行元素）

等原子組成では c_i - c_j = 0 のため拡張項は恒等的に消える。したがって
効果は非等原子組成合金に限定される。この予想も定量する。

出力: paper/rk_extension_metrics.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))
from hea_lattice_xgboost import (  # noqa: E402
    ALONSO_TABLE2, INDEPENDENT_TEST, KING_ATOMIC_VOLUMES,
)
from generate_all_figures import load_sqs_data  # noqa: E402

CSV_PATH = PAPER / "results_composition_dependence.csv"


def load_omega1(structure: str, endpoints: dict[str, float],
                endpoint_mode: str = "raw") -> dict:
    """組成依存解析の出力から無次元非対称係数 omega1 を読む。

    endpoint_mode="raw" は同一データセットの純元素端点、"curated" は
    King/MP置換後の端点で測った a1 を使う。純元素端点を置換すると
    dV(x) に x の一次項が混入して a1 が汚染されるため、既定は raw。
    """
    omega1 = {}
    with CSV_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["structure"] != structure:
                continue
            if row["endpoint"] != endpoint_mode:
                continue
            el_a, el_b = row["element_A"], row["element_B"]
            if el_a not in endpoints or el_b not in endpoints:
                continue
            denom = endpoints[el_a] + endpoints[el_b]
            omega1[(el_a, el_b)] = float(row["a1"]) / denom
    return omega1


def predict(comp: dict[str, float], struct: str, omega_sf: dict,
            omega1: dict, q: float, q1: float) -> float:
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements], dtype=float)
    fracs = fracs / fracs.sum()
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    n_auc = 4 if struct == "FCC" else 2
    n = len(elements)

    v_vegard = float(np.sum(fracs * vols))
    corr_sym = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = tuple(sorted([elements[i], elements[j]]))
            corr_sym += fracs[i] * fracs[j] * vols[j] * omega_sf.get(pair, 0.0)
    corr_asym = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = elements[i], elements[j]
            if a < b:
                pair, c_first, c_second = (a, b), fracs[i], fracs[j]
                v_sum = vols[i] + vols[j]
            else:
                pair, c_first, c_second = (b, a), fracs[j], fracs[i]
                v_sum = vols[i] + vols[j]
            w1 = omega1.get(pair)
            if w1 is None:
                continue
            corr_asym += c_first * c_second * (c_first - c_second) * v_sum * w1

    v_total = n_auc * (v_vegard + q * corr_sym + q1 * corr_asym)
    if v_total <= 0:
        return float("nan")
    return v_total ** (1.0 / 3.0)


def rmse(heas, omega_sf, omega1, q, q1) -> float:
    y = np.array([h["a_exp"] for h in heas])
    p = np.array([
        predict(h["comp"], h["struct"], omega_sf, omega1, q, q1) for h in heas
    ])
    return float(np.sqrt(np.mean((p - y) ** 2)))


def asym_term_magnitude(heas, omega_sf, omega1, q) -> dict:
    """拡張項が格子定数を動かす量（Å）を測る。"""
    shifts = []
    for h in heas:
        base = predict(h["comp"], h["struct"], omega_sf, omega1, q, 0.0)
        with_rk = predict(h["comp"], h["struct"], omega_sf, omega1, q, 1.0)
        shifts.append(abs(with_rk - base))
    shifts = np.array(shifts)
    return {
        "median_A": float(np.median(shifts)),
        "p90_A": float(np.percentile(shifts, 90)),
        "max_A": float(np.max(shifts)),
        "n": len(shifts),
    }


def is_equiatomic(comp: dict[str, float], tol: float = 0.02) -> bool:
    fracs = np.array(list(comp.values()), dtype=float)
    fracs = fracs / fracs.sum()
    return bool(np.max(np.abs(fracs - 1.0 / len(fracs))) <= tol)


def pair_coverage(heas, omega1) -> dict:
    total = 0
    covered = 0
    fully = 0
    for h in heas:
        els = sorted(h["comp"])
        pairs = [(els[i], els[j])
                 for i in range(len(els)) for j in range(i + 1, len(els))]
        total += len(pairs)
        hit = sum(1 for p in pairs if p in omega1)
        covered += hit
        fully += int(hit == len(pairs))
    return {
        "n_alloys": len(heas),
        "n_pairs": total,
        "n_pairs_covered": covered,
        "pair_coverage_frac": covered / total if total else None,
        "n_alloys_fully_covered": fully,
    }


def evaluate(struct: str, omega_sf: dict, omega1: dict) -> dict:
    train = [h for h in ALONSO_TABLE2 if h["struct"] == struct]
    test = [h for h in INDEPENDENT_TEST if h["struct"] == struct]

    def obj_q(params):
        return rmse(train, omega_sf, omega1, params[0], 0.0)

    res_q = minimize(obj_q, x0=[1.0], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-10})
    q_base = float(res_q.x[0])

    def obj_joint(params):
        return rmse(train, omega_sf, omega1, params[0], params[1])

    res_joint = minimize(obj_joint, x0=[q_base, 0.0], method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-10})
    q_j, q1_j = float(res_joint.x[0]), float(res_joint.x[1])

    def obj_q1_only(params):
        return rmse(train, omega_sf, omega1, 1.0, params[0])

    res_q1 = minimize(obj_q1_only, x0=[0.0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-10})
    q1_at_q1fixed = float(res_q1.x[0])

    non_eq_test = [h for h in test if not is_equiatomic(h["comp"])]
    eq_test = [h for h in test if is_equiatomic(h["comp"])]

    out = {
        "n_train": len(train),
        "n_test": len(test),
        "n_test_non_equiatomic": len(non_eq_test),
        "n_test_equiatomic": len(eq_test),
        "n_train_non_equiatomic": sum(
            1 for h in train if not is_equiatomic(h["comp"])
        ),
        "omega1_pairs_available": len(omega1),
        "coverage_train": pair_coverage(train, omega1),
        "coverage_test": pair_coverage(test, omega1),
        "q_baseline_opt": q_base,
        "rmse_train_baseline_qopt": rmse(train, omega_sf, omega1, q_base, 0.0),
        "rmse_test_baseline_qopt": rmse(test, omega_sf, omega1, q_base, 0.0),
        "rmse_train_baseline_q1": rmse(train, omega_sf, omega1, 1.0, 0.0),
        "rmse_test_baseline_q1": rmse(test, omega_sf, omega1, 1.0, 0.0),
        "q_joint": q_j,
        "q1_joint": q1_j,
        "rmse_train_joint": rmse(train, omega_sf, omega1, q_j, q1_j),
        "rmse_test_joint": rmse(test, omega_sf, omega1, q_j, q1_j),
        "q1_with_q_fixed_1": q1_at_q1fixed,
        "rmse_train_q1fixed": rmse(train, omega_sf, omega1, 1.0,
                                   q1_at_q1fixed),
        "rmse_test_q1fixed": rmse(test, omega_sf, omega1, 1.0,
                                  q1_at_q1fixed),
        "asym_term_shift_test_q1_unity": asym_term_magnitude(
            test, omega_sf, omega1, q_base
        ),
    }
    if non_eq_test:
        out["rmse_test_nonequi_baseline"] = rmse(
            non_eq_test, omega_sf, omega1, q_base, 0.0
        )
        out["rmse_test_nonequi_joint"] = rmse(
            non_eq_test, omega_sf, omega1, q_j, q1_j
        )
    if eq_test:
        out["rmse_test_equi_baseline"] = rmse(
            eq_test, omega_sf, omega1, q_base, 0.0
        )
        out["rmse_test_equi_joint"] = rmse(
            eq_test, omega_sf, omega1, q_j, q1_j
        )
    return out


def main() -> None:
    sqs = load_sqs_data()
    metrics = {}
    for mode, bcc_ep, fcc_ep in (
        ("raw", "pure_vol_raw", "fcc_pure_vol_raw"),
        ("curated", "pure_vol", "fcc_pure_vol"),
    ):
        bcc_endpoints = sqs.get(bcc_ep) or sqs["pure_vol"]
        fcc_endpoints = sqs.get(fcc_ep) or sqs["fcc_pure_vol"]
        metrics[f"BCC_sqs_dft_{mode}_endpoints"] = evaluate(
            "BCC", sqs["omega_dft"], load_omega1("BCC", bcc_endpoints, mode)
        )
        metrics[f"FCC_sqs_dft_{mode}_endpoints"] = evaluate(
            "FCC", sqs["fcc_omega_dft"],
            load_omega1("FCC", fcc_endpoints, mode)
        )
    with (PAPER / "rk_extension_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
