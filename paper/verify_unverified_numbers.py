#!/usr/bin/env python3
"""Item 7: make previously unverifiable paper numbers reproducible.

Covers:
  (7-1) 3-source Omega_sf consistency: r between sources, |Delta Omega| after merging
  (7-2) SQS overlay statistics on the Vegard parity figure
        (paper: BCC-SQS 515 pts RMSE 1.821, FCC-SQS 43 pts RMSE 0.802 A^3/atom)
Not covered (requires digitizing external tables, flagged for manual action):
  - Coreno-Alonso 2004 39-case comparison (0.76% vs 1.43%)
  - Alonso experimental Omega_sf (King 1966) test-set row (deleted in PR#439)
"""
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from generate_all_figures import EXCLUDE_ELEMENTS
from hea_lattice_xgboost import KING_ATOMIC_VOLUMES

DATA = HERE.parent / "data"
NAUC = {"B2": 2, "L12": 4}


def load_source_omegas(stype):
    """Per-source median Omega_sf per pair (50:50 for B2, 3:1 pooled for L12)."""
    out = {}
    for src in ["MP", "OQMD", "VASP"]:
        df = pd.read_csv(DATA / f"compounds_{src}_{stype}.csv")
        per_pair = {}
        for _, r in df.iterrows():
            a = r["lattice_constant"]
            elA, elB = r["element_A"], r["element_B"]
            if not (2 < a < 8) or elA == elB:
                continue
            if elA in EXCLUDE_ELEMENTS or elB in EXCLUDE_ELEMENTS:
                continue
            if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
                continue
            cA, cB = float(r.get("count_A", 1)), float(r.get("count_B", 1))
            v_act = a ** 3 / NAUC[stype]
            v_veg = (cA * KING_ATOMIC_VOLUMES[elA] +
                     cB * KING_ATOMIC_VOLUMES[elB]) / (cA + cB)
            pair = tuple(sorted([elA, elB]))
            per_pair.setdefault(pair, []).append((v_act - v_veg) / v_veg)
        out[src] = {p: float(np.median(v)) for p, v in per_pair.items()}
    return out


def main():
    print("=" * 78)
    print("[7-1] 3-source Omega_sf consistency (r and |Delta Omega|)")
    print("=" * 78)
    for stype in ["B2", "L12"]:
        srcs = load_source_omegas(stype)
        rs = []
        for s1, s2 in combinations(srcs, 2):
            common = sorted(set(srcs[s1]) & set(srcs[s2]))
            if len(common) < 3:
                continue
            x = np.array([srcs[s1][p] for p in common])
            y = np.array([srcs[s2][p] for p in common])
            r = float(np.corrcoef(x, y)[0, 1])
            rs.append(r)
            print(f"  {stype}: {s1} vs {s2}: r={r:.3f} (n={len(common)})")
        # |Delta Omega|: change from single-source value to merged median
        merged = {}
        for src, om in srcs.items():
            for p, v in om.items():
                merged.setdefault(p, []).append(v)
        deltas = []
        for p, vals in merged.items():
            if len(vals) >= 2:
                m = float(np.median(vals))
                deltas += [abs(v - m) for v in vals]
        print(f"  {stype}: mean |Delta Omega| (source -> merged median) = "
              f"{np.mean(deltas):.4f}; pairwise r range "
              f"[{min(rs):.3f}, {max(rs):.3f}]")

    print()
    print("=" * 78)
    print("[7-2] SQS overlay statistics on Vegard parity (volume space)")
    print("=" * 78)
    sqs = pd.read_csv(DATA / "sqs_results.csv")
    for root, label in [("BCC_SQS", "BCC-SQS"), ("FCC_SQS", "FCC-SQS")]:
        # Keep the cell size used by the paper's overlay statistics:
        # BCC 16-atom SQS (8:8) and FCC 32-atom SQS (16:16).
        expected_count = 8 if root == "BCC_SQS" else 16
        sub = sqs[(sqs["structure_root"] == root) & (sqs["status"] == "OK")
                  & (sqs["relax_converged"] == "yes")]
        pts = []
        for _, r in sub.iterrows():
            m = re.match(r"([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)$", r["dir"])
            if not m:
                continue
            ea, na, eb, nb = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            if ea == eb:
                continue
            if ea in EXCLUDE_ELEMENTS or eb in EXCLUDE_ELEMENTS:
                continue
            if ea not in KING_ATOMIC_VOLUMES or eb not in KING_ATOMIC_VOLUMES:
                continue
            if na != expected_count or nb != expected_count:
                continue  # match the canonical SQS cell size used in the paper
            v_act = float(r["volume_A3"]) / float(r["natoms"])
            v_veg = (na * KING_ATOMIC_VOLUMES[ea] +
                     nb * KING_ATOMIC_VOLUMES[eb]) / (na + nb)
            pts.append((v_veg, v_act))
        v_veg = np.array([p[0] for p in pts])
        v_act = np.array([p[1] for p in pts])
        res = v_act - v_veg
        rmse = float(np.sqrt(np.mean(res ** 2)))
        ss_tot = float(np.sum((v_act - v_act.mean()) ** 2))
        r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot
        print(f"  {label}: n={len(pts)}, RMSE(vs Vegard)={rmse:.3f} A^3/atom, "
              f"R^2={r2:.3f}")
    print()
    print("  Paper claims: BCC-SQS 515 pts / RMSE 1.821 / R^2 0.762; "
          "FCC-SQS 43 pts / RMSE 0.802")
    print("  -> NOT reproducible under any tested filter combination "
          "(all / converged / 50:50 / no-exclusion).")
    print("     Recommended canonical stats: values printed above "
          "(converged, 50:50, exclusions applied).")
    print()
    print("[7-3] NOT reproducible from bundled data (manual digitization needed):")
    print("  - Coreno-Alonso 2004 39-case comparison (0.76% vs 1.43%)")
    print("  - Alonso experimental Omega_sf (King 1966 table) test-set evaluation")


if __name__ == "__main__":
    main()
