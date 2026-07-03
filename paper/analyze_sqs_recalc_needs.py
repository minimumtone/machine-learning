#!/usr/bin/env python3
"""Analyze current SQS dataset to identify what needs recalculation.

Outputs:
  - BCC/FCC pure elements replaced by King/MP (reliability check failures)
  - SQS pair coverage gaps vs B2 pairs used in HEA test predictions
  - Missing pairs required by the independent test set
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_all_figures import (
    load_sqs_data, load_compounds, compute_omega_sf_pairwise,
    EXCLUDE_ELEMENTS, KING_ATOMIC_VOLUMES,
)
from hea_lattice_xgboost import ALONSO_TABLE2, INDEPENDENT_TEST
from itertools import combinations

sqs = load_sqs_data()

print("=" * 70)
print("[1] BCC pure elements replaced (SQS deviates >3% from King)")
print("=" * 70)
replaced_bcc = []
for el, v_raw in sorted(sqs["pure_vol_raw"].items()):
    v_used = sqs["pure_vol"][el]
    if abs(v_raw - v_used) > 1e-6:
        v_king = KING_ATOMIC_VOLUMES.get(el)
        pct = (v_raw - v_king) / v_king * 100 if v_king else float("nan")
        replaced_bcc.append(el)
        print(f"  {el:3s}: SQS {v_raw:8.3f} -> used {v_used:8.3f} "
              f"(King {v_king:8.3f}, SQS-King {pct:+.1f}%)")
print(f"  Total: {len(replaced_bcc)} elements: {', '.join(replaced_bcc)}")

print()
print("=" * 70)
print("[2] FCC pure elements replaced")
print("=" * 70)
replaced_fcc = []
for el, v_raw in sorted(sqs["fcc_pure_vol_raw"].items()):
    v_used = sqs["fcc_pure_vol"][el]
    if abs(v_raw - v_used) > 1e-6:
        v_king = KING_ATOMIC_VOLUMES.get(el)
        pct = (v_raw - v_king) / v_king * 100 if v_king else float("nan")
        replaced_fcc.append(el)
        print(f"  {el:3s}: SQS {v_raw:8.3f} -> used {v_used:8.3f} "
              f"(King {v_king:8.3f}, SQS-King {pct:+.1f}%)")
print(f"  Total: {len(replaced_fcc)} elements: {', '.join(replaced_fcc)}")

print()
print("=" * 70)
print("[3] SQS BCC pair coverage vs independent test requirements")
print("=" * 70)
sqs_pairs = set(sqs["omega_dft"].keys()) | set(sqs["omega_king"].keys())
test_bcc = [h for h in INDEPENDENT_TEST if h["struct"] == "BCC"]
train_bcc = [h for h in ALONSO_TABLE2 if h["struct"] == "BCC"]
missing_test = set()
for h in test_bcc + train_bcc:
    els = [e for e in h["comp"] if e not in EXCLUDE_ELEMENTS]
    for pair in combinations(sorted(els), 2):
        if pair not in sqs_pairs:
            missing_test.add(pair)
if missing_test:
    for p in sorted(missing_test):
        print(f"  MISSING (needed by calib/test BCC HEA): {p[0]}-{p[1]}")
else:
    print("  No missing pairs for calibration/test BCC HEAs")

print()
print("=" * 70)
print("[4] SQS pair coverage vs B2 pairs (systematic gap check)")
print("=" * 70)
all_df = load_compounds()
ob2, ol12 = compute_omega_sf_pairwise(all_df)
b2_pairs = set(ob2.keys())
missing_vs_b2 = sorted(b2_pairs - sqs_pairs)
print(f"  B2 pairs: {len(b2_pairs)}, SQS pairs: {len(sqs_pairs)}, "
      f"missing in SQS: {len(missing_vs_b2)}")
import numpy as np
if missing_vs_b2:
    mags_missing = [abs(ob2[p]) for p in missing_vs_b2]
    mags_covered = [abs(ob2[p]) for p in sorted(b2_pairs & sqs_pairs)]
    print(f"  |Omega_B2| missing pairs:  mean {np.mean(mags_missing):.4f}, "
          f"median {np.median(mags_missing):.4f}")
    print(f"  |Omega_B2| covered pairs:  mean {np.mean(mags_covered):.4f}, "
          f"median {np.median(mags_covered):.4f}")
    print("  Missing pairs (sorted by |Omega_B2| desc, top 30):")
    for p in sorted(missing_vs_b2, key=lambda p: -abs(ob2[p]))[:30]:
        print(f"    {p[0]}-{p[1]}: Omega_B2 = {ob2[p]:+.4f}")

print()
print("=" * 70)
print("[4b] SQS pure elements vs MP-BCC DFT volumes (same-functional check)")
print("=" * 70)
import pandas as pd
mp_bcc = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "mp_pure_elements_bcc.csv")
mp_bcc_vol = dict(zip(mp_bcc["element"], mp_bcc["volume_per_atom"]))
rows_dev = []
for el, v_sqs in sorted(sqs["pure_vol_raw"].items()):
    v_mp = mp_bcc_vol.get(el)
    if v_mp:
        rows_dev.append((el, v_sqs, v_mp, (v_sqs - v_mp) / v_mp * 100))
rows_dev.sort(key=lambda r: -abs(r[3]))
print("  el   SQS(A3)   MP(A3)   dev%  (sorted by |dev|)")
for el, v_sqs, v_mp, dev in rows_dev:
    flag = " <-- >3%" if abs(dev) > 3 else ""
    print(f"  {el:3s} {v_sqs:8.3f} {v_mp:8.3f} {dev:+6.1f}{flag}")

print()
print("=" * 70)
print("[4c] SQS 1:1 pair volumes vs MP/OQMD B2 volumes (deviation ranking)")
print("=" * 70)
data_dir = Path(__file__).resolve().parent.parent / "data"
b2_src = {}
for src in ["MP", "OQMD"]:
    df = pd.read_csv(data_dir / f"compounds_{src}_B2.csv")
    for _, r in df.iterrows():
        a = r["lattice_constant"]
        if not (2 < a < 8):
            continue
        pair = tuple(sorted([r["element_A"], r["element_B"]]))
        b2_src.setdefault(pair, {}).setdefault(src, []).append(a**3 / 2)
pair_dev = []
for pair, a_sqs in sqs["sqs_a"].items():
    v_sqs = a_sqs**3 / 2
    refs = b2_src.get(pair)
    if not refs:
        continue
    import numpy as _np
    vals = [ _np.median(v) for v in refs.values() ]
    v_ref = float(_np.mean(vals))
    pair_dev.append((pair, v_sqs, v_ref, (v_sqs - v_ref) / v_ref * 100,
                     "+".join(sorted(refs))))
pair_dev.sort(key=lambda r: -abs(r[3]))
n_over5 = sum(1 for r in pair_dev if abs(r[3]) > 5)
n_over3 = sum(1 for r in pair_dev if abs(r[3]) > 3)
print(f"  {len(pair_dev)} SQS pairs have MP/OQMD B2 reference; "
      f"|dev|>3%: {n_over3}, |dev|>5%: {n_over5}")
print("  Note: SQS(random) vs B2(ordered) differ physically; large |dev| is a flag.")
print("  pair        V_SQS   V_B2ref   dev%   ref  (top 40 by |dev|)")
for pair, v_sqs, v_ref, dev, srcs in pair_dev[:40]:
    print(f"  {pair[0]}-{pair[1]:<4s} {v_sqs:8.3f} {v_ref:8.3f} {dev:+7.1f}   {srcs}")
with open(Path(__file__).resolve().parent / "sqs_vs_mp_oqmd_deviation.csv", "w") as f:
    f.write("pair,V_sqs_A3,V_b2ref_A3,dev_pct,ref_sources\n")
    for pair, v_sqs, v_ref, dev, srcs in pair_dev:
        f.write(f"{pair[0]}-{pair[1]},{v_sqs:.4f},{v_ref:.4f},{dev:.2f},{srcs}\n")
print("  Full ranking written to paper/sqs_vs_mp_oqmd_deviation.csv")

print()
print("=" * 70)
print("[5] Elements present in SQS pure set")
print("=" * 70)
print(f"  BCC pure ({len(sqs['pure_vol_raw'])}): "
      f"{', '.join(sorted(sqs['pure_vol_raw']))}")
print(f"  FCC pure ({len(sqs['fcc_pure_vol_raw'])}): "
      f"{', '.join(sorted(sqs['fcc_pure_vol_raw']))}")
