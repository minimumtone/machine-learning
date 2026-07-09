#!/usr/bin/env python3
"""Alonso experimental Omega_sf model (King 1966 Table II) test-set evaluation.

Restores the previously unverifiable "Alonso volume size-factor" row (review
item B1 / verification item 7) using digitized experimental data:
  data/king_1966_table2_size_factors.csv     (directional volume size factors)
  data/king_1966_table3_atomic_volumes.csv   (atomic volumes / Seitz radii)
  data/alonso_table3_volume_size_factors.csv (Alonso 2022 Table 3 supplement)

Duplicate directional (solvent, solute) entries in King Table II (different
Cmax concentration ranges) are averaged. Rows whose lsf is inconsistent with
Dsf (|lsf - ((1+Dsf)^(1/3)-1)| > 0.5%) are excluded; rows verified against
the original table pages (Ir-Mn, Ir-Ta, Nb-W, Ni-Zn, Pd-Hg, Pd-Mn, Re-Os,
Re-Pt, Rh-Pt, Tc-Ir, U-Pu) match the original print, so the inconsistency is
a typographical error in King 1966 itself. Fourteen rows whose solvent was
misassigned during digitization (right-column bleed in the two-column layout,
e.g. Pd-Pd -> Rh-Pd/Ru-Pd, Ta-Ta -> V-Ta, Sr-Pu -> U-Pu) were corrected
against the original pages. Alonso Table 3 values supplement directional
pairs absent from King Table II (both directions kept).

Model: Alonso Eq.10 with directional experimental Omega_sf,j/i = Dsf/100
(solute j in solvent i); missing pairs fall back to the reversed direction,
then to 0 (Vegard). Evaluated both with q=1 (uncalibrated, original Alonso)
and with q_BCC/q_FCC calibrated on the 64-HEA calibration set.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from hea_lattice_xgboost import (
    KING_ATOMIC_VOLUMES, ALONSO_TABLE2, INDEPENDENT_TEST, compute_vegard,
)

DATA = HERE.parent / "data"


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def predict_alonso(comp, struct, omega_dir, q=1.0):
    """Alonso Eq.10 with directional Omega_sf (solvent i, solute j)."""
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements], dtype=float)
    fracs = fracs / fracs.sum()
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    n_auc = 4 if struct == "FCC" else 2
    v_veg = float(np.sum(fracs * vols))
    corr = 0.0
    for i, ei in enumerate(elements):
        for j, ej in enumerate(elements):
            if i == j:
                continue
            om = omega_dir.get((ei, ej))
            if om is None:
                om = omega_dir.get((ej, ei), 0.0)
            corr += fracs[i] * fracs[j] * vols[j] * om
    v = n_auc * (v_veg + q * corr)
    if v <= 0:
        return compute_vegard(comp, struct)
    return v ** (1 / 3)


def eval_set(heas, omega_dir, qb, qf):
    y = np.array([h["a_exp"] for h in heas])
    p = np.array([predict_alonso(h["comp"], h["struct"], omega_dir,
                                 qb if h["struct"] == "BCC" else qf)
                  for h in heas])
    return rmse(p, y)


def main():
    t2 = pd.read_csv(DATA / "king_1966_table2_size_factors.csv")

    self_pairs = t2[t2.Solvent == t2.Solute]
    if len(self_pairs):
        print(f"WARNING: {len(self_pairs)} self-pair rows excluded "
              f"(solute likely misdigitized):")
        for r in self_pairs.itertuples():
            print(f"  line: {r.Solvent},{r.Solute},{r.Cmax_at_percent},"
                  f"{r.Dsf_percent},...")
        t2 = t2[t2.Solvent != t2.Solute]
        print()

    # internal-consistency check: lsf should equal (1+Dsf)^(1/3)-1
    lsf_pred = 100 * ((1 + t2.Dsf_percent / 100.0) ** (1 / 3) - 1)
    bad = t2[(t2.lsf_percent - lsf_pred).abs() > 0.5]
    if len(bad):
        print(f"WARNING: {len(bad)} rows with lsf inconsistent with Dsf "
              f"(possible digitization errors):")
        for r in bad.itertuples():
            print(f"  {r.Solvent}-{r.Solute}: Dsf={r.Dsf_percent} "
                  f"lsf={r.lsf_percent} (expected "
                  f"{lsf_pred[r.Index]:.2f})")
        print("  -> these rows are EXCLUDED from the model "
              "(pending re-check against the original table)")
        t2 = t2.drop(bad.index)
        print()

    # average duplicate directional entries (different Cmax ranges)
    grouped = t2.groupby(["Solvent", "Solute"])["Dsf_percent"]
    dups = grouped.filter(lambda g: len(g) > 1)
    if len(dups):
        print("Duplicate (solvent, solute) entries averaged:")
        for (sv, sl), g in t2.groupby(["Solvent", "Solute"]):
            if len(g) > 1:
                vals = ", ".join(f"{v:+.2f}" for v in g.Dsf_percent)
                print(f"  {sv}-{sl}: [{vals}] -> {g.Dsf_percent.mean():+.2f}")
        print()
    omega_dir = {p: v / 100.0 for p, v in grouped.mean().items()}
    n_king = len(omega_dir)

    # Alonso (2022) Table 3 supplement: only pairs absent from King Table II
    ta = pd.read_csv(DATA / "alonso_table3_volume_size_factors.csv")
    n_added = 0
    for r in ta.itertuples():
        key = (r.Solvent, r.Solute)
        if key not in omega_dir:
            omega_dir[key] = r.Omega_f_percent / 100.0
            n_added += 1
    print(f"King Table II: {len(t2)} rows, {n_king} directional pairs; "
          f"Alonso Table 3: {len(ta)} rows, {n_added} new pairs added; "
          f"total {len(omega_dir)} directional, "
          f"{len({tuple(sorted(p)) for p in omega_dir})} unique pairs")

    # Coverage of pairs needed by calibration + test HEAs
    need = set()
    for h in ALONSO_TABLE2 + INDEPENDENT_TEST:
        need |= set(combinations(sorted(h["comp"]), 2))
    sym = {tuple(sorted(p)) for p in omega_dir}
    print(f"HEA-required pairs: {len(need)}, covered: {len(need & sym)} "
          f"({100 * len(need & sym) / len(need):.0f}%)")
    print(f"missing: {sorted(need - sym)}")
    print()

    # q=1 (uncalibrated) and calibrated q_BCC/q_FCC on the 64-HEA set
    def calib_q(struct):
        heas = [h for h in ALONSO_TABLE2 if h["struct"] == struct]
        y = np.array([h["a_exp"] for h in heas])

        def f(q):
            p = [predict_alonso(h["comp"], h["struct"], omega_dir, q)
                 for h in heas]
            return rmse(p, y)
        return float(minimize_scalar(f, bounds=(-5, 5), method="bounded").x)

    qb, qf = calib_q("BCC"), calib_q("FCC")
    print(f"calibrated q_BCC={qb:+.3f}, q_FCC={qf:+.3f}")
    print()

    test_b = [h for h in INDEPENDENT_TEST if h["struct"] == "BCC"]
    test_f = [h for h in INDEPENDENT_TEST if h["struct"] == "FCC"]
    header = f"  {'model':30s} {'all(31)':>8s} {'BCC(17)':>8s} {'FCC(14)':>8s}"
    print(header)
    for label, args in [
        ("Vegard (King volumes)", (0.0, 0.0)),
        ("Alonso-King+T3 Omega_sf, q=1", (1.0, 1.0)),
        ("Alonso-King+T3, q calib", (qb, qf)),
    ]:
        r_all = eval_set(INDEPENDENT_TEST, omega_dir, *args)
        r_b = eval_set(test_b, omega_dir, *args)
        r_f = eval_set(test_f, omega_dir, *args)
        print(f"  {label:30s} {r_all:8.4f} {r_b:8.4f} {r_f:8.4f}")

    print()
    print("Table III atomic volumes vs KING_ATOMIC_VOLUMES in code:")
    t3 = pd.read_csv(DATA / "king_1966_table3_atomic_volumes.csv")
    t3v = dict(zip(t3.Element, t3.Atomic_volume_A3))
    for el, v in sorted(KING_ATOMIC_VOLUMES.items()):
        v3 = t3v.get(el)
        if v3 is None:
            print(f"  {el:3s}: code {v:7.3f}  NOT in Table III")
        elif abs(v3 - v) / v > 0.005:
            print(f"  {el:3s}: code {v:7.3f}  TableIII {v3:7.3f} "
                  f"({100 * (v3 - v) / v:+.1f}%)")
    print("  (all other elements agree within 0.5%)")


if __name__ == "__main__":
    main()
