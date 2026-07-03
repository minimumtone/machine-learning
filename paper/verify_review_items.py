#!/usr/bin/env python3
"""Reviewer-verification computations (items 1a/1b/1c, 3, 5, 8).

Run:  python paper/verify_review_items.py
All results printed and saved to paper/verify_review_items_report.txt implicitly
via tee by the caller if desired.
"""
import csv
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from generate_all_figures import (
    load_sqs_data, load_compounds, compute_omega_sf_pairwise,
    optimize_gamma, predict_heas,
    EXCLUDE_ELEMENTS, SQS_FILE,
)
from hea_lattice_xgboost import (
    KING_ATOMIC_VOLUMES, ALONSO_TABLE2, INDEPENDENT_TEST,
    compute_eq10_scaled, compute_vegard, compute_delta_r, compute_delta_sf,
    PAULING_EN, VEC,
)

import pandas as pd

BCC_TRAIN = [h for h in ALONSO_TABLE2 if h["struct"] == "BCC"]
BCC_TEST = [h for h in INDEPENDENT_TEST if h["struct"] == "BCC"]


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def eval_omega(omega, q, heas):
    y = [h["a_exp"] for h in heas]
    p = [compute_eq10_scaled(h["comp"], h["struct"], omega, q) for h in heas]
    return rmse(p, y)


def optimize_q(omega, heas):
    res = minimize_scalar(lambda q: eval_omega(omega, q, heas),
                          bounds=(-5, 5), method="bounded")
    return float(res.x), float(res.fun)


def load_sqs_rows():
    with open(SQS_FILE) as f:
        return list(csv.DictReader(f))


def build_omega_from_pure(rows, pure_vol):
    """Rebuild BCC SQS omega dict given a pure-element volume dict."""
    omega = {}
    for r in rows:
        if r["status"] != "OK" or r.get("relax_converged", "") != "yes":
            continue
        if r["lattice_type"] != "bcc":
            continue
        prs = re.findall(r"([A-Z][a-z]?)(\d+)", r["dir"])
        if len(prs) != 2:
            continue
        el1, n1 = prs[0][0], int(prs[0][1])
        el2, n2 = prs[1][0], int(prs[1][1])
        if n1 != 8 or n2 != 8 or el1 == el2:
            continue
        if el1 in EXCLUDE_ELEMENTS or el2 in EXCLUDE_ELEMENTS:
            continue
        if el1 not in KING_ATOMIC_VOLUMES or el2 not in KING_ATOMIC_VOLUMES:
            continue
        try:
            a = float(r["a_bcc_A"])
        except (ValueError, KeyError):
            continue
        if not (2.0 < a < 8.0):
            continue
        pair = tuple(sorted([el1, el2]))
        if pair[0] in pure_vol and pair[1] in pure_vol:
            v_ref = 0.5 * pure_vol[pair[0]] + 0.5 * pure_vol[pair[1]]
            omega[pair] = (a ** 3 / 2.0 - v_ref) / v_ref
    return omega


def report_variant(label, omega, extra=""):
    q_opt, rmse_tr = optimize_q(omega, BCC_TRAIN)
    r_test_qopt = eval_omega(omega, q_opt, BCC_TEST)
    r_test_q1 = eval_omega(omega, 1.0, BCC_TEST)
    r_veg = eval_omega({}, 0.0, BCC_TEST)
    print(f"  {label:34s} q_opt={q_opt:+.3f}  "
          f"test(q_opt)={r_test_qopt:.4f}  test(q=1)={r_test_q1:.4f}  "
          f"[Vegard {r_veg:.4f}] {extra}")
    return q_opt, r_test_qopt, r_test_q1


def main():
    sqs = load_sqs_data()
    rows = load_sqs_rows()
    pure_raw = sqs["pure_vol_raw"]
    pure_hybrid = sqs["pure_vol"]
    mp_df = pd.read_csv(HERE.parent / "data" / "mp_pure_elements_bcc.csv")
    mp_vol = dict(zip(mp_df["element"], mp_df["volume_per_atom"]))

    print("=" * 78)
    print("[1a] q re-optimization: raw SQS vs hybrid vs all-King pure volumes")
    print("=" * 78)
    variants = {
        "(1) raw SQS (no replacement)": pure_raw,
        "(2) hybrid (current, 16 replaced)": pure_hybrid,
        "(3) all King": {el: KING_ATOMIC_VOLUMES[el] for el in pure_raw
                         if el in KING_ATOMIC_VOLUMES},
        "(4) all MP-BCC DFT": {el: mp_vol[el] for el in pure_raw
                               if el in mp_vol},
    }
    for label, pv in variants.items():
        omega = build_omega_from_pure(rows, pv)
        report_variant(label, omega, extra=f"(n_pairs={len(omega)})")

    print()
    print("=" * 78)
    print("[1b] Non-circular replacement: criterion = |SQS-MP| > 3% (same functional)")
    print("=" * 78)
    pv_mpcrit = {}
    replaced = []
    for el, v in pure_raw.items():
        v_mp = mp_vol.get(el)
        if v_mp and abs(v - v_mp) / v_mp * 100 > 3.0:
            pv_mpcrit[el] = v_mp
            replaced.append(f"{el}({(v - v_mp) / v_mp * 100:+.1f}%)")
        else:
            pv_mpcrit[el] = v
    print(f"  replaced ({len(replaced)}): {', '.join(replaced)}")
    omega = build_omega_from_pure(rows, pv_mpcrit)
    report_variant("(5) MP-criterion hybrid", omega)

    print()
    print("=" * 78)
    print("[1c] Cr/Fe/Mn individual sensitivity (others = current hybrid)")
    print("=" * 78)
    for el in ["Cr", "Fe", "Mn"]:
        for src, val in [("SQS", pure_raw.get(el)),
                         ("MP", mp_vol.get(el)),
                         ("King", KING_ATOMIC_VOLUMES.get(el))]:
            if val is None:
                continue
            pv = dict(pure_hybrid)
            pv[el] = val
            omega = build_omega_from_pure(rows, pv)
            report_variant(f"{el} <- {src} ({val:.3f} A3)", omega)

    print()
    print("=" * 78)
    print("[3a] Test-set independence: exclude 3 compositions overlapping calibration")
    print("=" * 78)
    all_df = load_compounds()
    ob2, ol12 = compute_omega_sf_pairwise(all_df)
    gb, gf = optimize_gamma(ALONSO_TABLE2, ob2, ol12)

    def comp_key(comp):
        tot = sum(comp.values())
        return tuple(sorted((e, round(v / tot, 3)) for e, v in comp.items()))

    calib_keys = {comp_key(h["comp"]) for h in ALONSO_TABLE2}
    overlap = [h for h in INDEPENDENT_TEST if comp_key(h["comp"]) in calib_keys]
    kept = [h for h in INDEPENDENT_TEST if comp_key(h["comp"]) not in calib_keys]
    def fname(h):
        return h.get("name") or "".join(sorted(h["comp"]))
    print(f"  overlapping compositions ({len(overlap)}): "
          f"{[fname(h) for h in overlap]}")
    for label, subset in [("full test (31)", INDEPENDENT_TEST),
                          (f"non-overlap ({len(kept)})", kept)]:
        y = np.array([h["a_exp"] for h in subset])
        a_veg = np.array([compute_vegard(h["comp"], h["struct"]) for h in subset])
        a_ss = predict_heas(subset, ob2, ol12, gb, gf)
        rv, rs = rmse(a_veg, y), rmse(a_ss, y)
        print(f"  {label:20s} Vegard={rv:.4f}  model={rs:.4f}  "
              f"improvement={100 * (1 - rs / rv):.1f}%")

    print()
    print("=" * 78)
    print("[3b] Leave-one-element-out (BCC): q re-optimized w/o element, "
          "eval on test alloys containing it")
    print("=" * 78)
    test_els = sorted({e for h in BCC_TEST for e in h["comp"]})
    for el in test_els:
        calib_wo = [h for h in BCC_TRAIN if el not in h["comp"]]
        test_with = [h for h in BCC_TEST if el in h["comp"]]
        if not test_with or len(calib_wo) < 5:
            continue
        gb_wo, _ = optimize_gamma(calib_wo + [h for h in ALONSO_TABLE2
                                              if h["struct"] == "FCC"], ob2, ol12)
        y = np.array([h["a_exp"] for h in test_with])
        a_ss = predict_heas(test_with, ob2, ol12, gb_wo, gf)
        a_veg = np.array([compute_vegard(h["comp"], h["struct"])
                          for h in test_with])
        print(f"  w/o {el:2s}: calib {len(calib_wo):2d}/{len(BCC_TRAIN)}, "
              f"test n={len(test_with):2d}, q_BCC={gb_wo:+.3f}, "
              f"Vegard={rmse(a_veg, y):.4f}, model={rmse(a_ss, y):.4f}")

    print()
    print("=" * 78)
    print("[5] L1_2 asymmetry: directional (A3B vs B3A kept separate) FCC impact")
    print("=" * 78)
    # Build orientation-resolved L12 omegas
    lo, hi = {}, {}
    per_orient = {}
    sub = all_df[all_df["stype"] == "L12"]
    for _, r in sub.iterrows():
        elA, elB = r.get("element_A", ""), r.get("element_B", "")
        a = r.get("lattice_constant", 0)
        if not (2 < a < 8) or elA == elB:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue
        cA, cB = r.get("count_A", 3), r.get("count_B", 1)
        v_act = a ** 3 / 4
        v_veg = (cA * KING_ATOMIC_VOLUMES[elA] + cB * KING_ATOMIC_VOLUMES[elB]) / (cA + cB)
        maj = elA if cA >= cB else elB
        pair = tuple(sorted([elA, elB]))
        per_orient.setdefault((pair, maj), []).append((v_act - v_veg) / v_veg)
    pair_orients = {}
    for (pair, maj), vals in per_orient.items():
        pair_orients.setdefault(pair, {})[maj] = float(np.median(vals))
    n_both = sum(1 for v in pair_orients.values() if len(v) == 2)
    asym = [abs(vals[0] - vals[1]) for v in pair_orients.values()
            if len(v) == 2 for vals in [list(v.values())]]
    print(f"  L1_2 pairs with both orientations: {n_both}/{len(pair_orients)}; "
          f"|Omega(A3B)-Omega(B3A)| mean={np.mean(asym):.4f}, "
          f"max={np.max(asym):.4f}")
    for pair, ov in pair_orients.items():
        vals = list(ov.values())
        lo[pair] = min(vals)
        hi[pair] = max(vals)
    fcc_test = [h for h in INDEPENDENT_TEST if h["struct"] == "FCC"]
    y_f = np.array([h["a_exp"] for h in fcc_test])
    p_sym = predict_heas(fcc_test, ob2, ol12, gb, gf)
    p_lo = predict_heas(fcc_test, ob2, lo, gb, gf)
    p_hi = predict_heas(fcc_test, ob2, hi, gb, gf)
    print(f"  FCC test RMSE: symmetric(median)={rmse(p_sym, y_f):.4f}, "
          f"min-orient={rmse(p_lo, y_f):.4f}, max-orient={rmse(p_hi, y_f):.4f}")
    print(f"  max |prediction difference| across orientations: "
          f"{np.max(np.abs(p_hi - p_lo)):.4f} A "
          f"(mean {np.mean(np.abs(p_hi - p_lo)):.4f} A)")

    print()
    print("=" * 78)
    print("[8] Fair ML comparison: GradientBoosting on RESIDUALS (LOO-CV), "
          "and direct with Vegard feature")
    print("=" * 78)
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import LeaveOneOut

    y_train = np.array([h["a_exp"] for h in ALONSO_TABLE2])
    a_ss_tr = predict_heas(ALONSO_TABLE2, ob2, ol12, gb, gf)
    a_veg_tr = np.array([compute_vegard(h["comp"], h["struct"])
                         for h in ALONSO_TABLE2])
    residuals = y_train - a_ss_tr
    n = len(ALONSO_TABLE2)
    X = np.zeros((n, 6))
    for i, h in enumerate(ALONSO_TABLE2):
        comp = h["comp"]
        els = list(comp.keys())
        fr = np.array([comp[e] for e in els])
        fr = fr / fr.sum()
        vecs = np.array([VEC.get(e, 0) for e in els])
        ens = np.array([PAULING_EN.get(e, 0) for e in els])
        om = ob2 if h["struct"] == "BCC" else ol12
        X[i] = [np.dot(fr, vecs), compute_delta_r(comp),
                compute_delta_sf(comp, om), np.dot(fr, ens),
                len(els), np.std(vecs)]

    loo = LeaveOneOut()
    print(f"  physics model train RMSE: {rmse(a_ss_tr, y_train):.4f}")
    # Ridge residual (existing, for reference)
    pred_r = np.zeros(n)
    for tr, te in loo.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], residuals[tr])
        pred_r[te] = m.predict(X[te])
    print(f"  Ridge residual LOO-CV:    {rmse(a_ss_tr + pred_r, y_train):.4f}")
    # GB residual (fair comparison)
    pred_g = np.zeros(n)
    for tr, te in loo.split(X):
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                      random_state=42).fit(X[tr], residuals[tr])
        pred_g[te] = m.predict(X[te])
    print(f"  GB residual LOO-CV:       {rmse(a_ss_tr + pred_g, y_train):.4f}")
    # GB direct WITH Vegard feature
    Xv = np.hstack([X, a_veg_tr[:, None]])
    pred_d = np.zeros(n)
    for tr, te in loo.split(Xv):
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                      random_state=42).fit(Xv[tr], y_train[tr])
        pred_d[te] = m.predict(Xv[te])
    print(f"  GB direct (+Vegard feat): {rmse(pred_d, y_train):.4f}")
    # Ridge direct with Vegard feature (linear baseline)
    pred_rd = np.zeros(n)
    for tr, te in loo.split(Xv):
        m = Ridge(alpha=1.0).fit(Xv[tr], y_train[tr])
        pred_rd[te] = m.predict(Xv[te])
    print(f"  Ridge direct (+Vegard):   {rmse(pred_rd, y_train):.4f}")


if __name__ == "__main__":
    main()
