#!/usr/bin/env python3
"""
Focused recalculation: extract key numbers from VASP-integrated dataset
for LaTeX manuscript update.

Produces:
  1. Tab.1 (DFT data counts) — correct MP/OQMD/VASP breakdowns
  2. Ω_sf coverage (B2, L12) — filtered (>=2 entries)
  3. Training set results (64 HEA) — γ/q optimization
  4. Independent test results (20 HEA) — key: BCC Vegard degeneracy check
  5. Additive decomposition stats
  6. 905-pair recount
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# Import from main script
from hea_lattice_xgboost import (
    KING_ATOMIC_VOLUMES, ALONSO_TABLE2, INDEPENDENT_TEST,
    load_compound_data, compute_structure_specific_omega_sf,
    compute_dft_omega_sf, compute_vegard, compute_eq10_scaled,
    compute_eq10_dft, estimate_noise_floor,
    build_omega_sf_ml_model, fill_missing_omega_sf,
)

# Exclude 4f rare earths + Y (GGA-PBE unreliable for 4f localized electrons)
# Also exclude Si, Ge, B (non-BCC/FCC stable structure)
_EXCLUDE = {
    "Gd", "Ce", "La", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",  # 4f RE
    "Y",   # similar RE behavior
    "Si", "Ge", "B",  # non-BCC/FCC stable structure
    "Li", "P",  # non-metals
}
TARGET_ELEMENTS = sorted([
    e for e in KING_ATOMIC_VOLUMES.keys()
    if e not in _EXCLUDE
])

OUTDIR = Path("recalc_output")
OUTDIR.mkdir(exist_ok=True)

def rmse(err):
    return np.sqrt(np.mean(err**2))

print("=" * 70)
print("RECALCULATION: VASP-integrated dataset audit")
print("=" * 70)

# =====================================================================
# 1. Data counts
# =====================================================================
print("\n[1] Data counts from actual CSV files...")

sources = {
    "MP_B2": "four_case_output/figures/compounds_MP_B2.csv",
    "MP_L12": "four_case_output/figures/compounds_MP_L12.csv",
    "OQMD_B2": "four_case_output/figures/compounds_OQMD_B2.csv",
    "OQMD_L12": "four_case_output/figures/compounds_OQMD_L12.csv",
    "VASP_B2": "data/compounds_VASP_B2.csv",
    "VASP_L12": "data/compounds_VASP_L12.csv",
}

counts = {}
for key, path in sources.items():
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        counts[key] = len(df)
        print(f"  {key}: {len(df)} rows")
    else:
        counts[key] = 0
        print(f"  {key}: FILE NOT FOUND")

print("\n  === Tab.1 correct values ===")
print(f"  MP:   B2={counts['MP_B2']}, L12={counts['MP_L12']}, total={counts['MP_B2']+counts['MP_L12']}")
print(f"  OQMD: B2={counts['OQMD_B2']}, L12={counts['OQMD_L12']}, total={counts['OQMD_B2']+counts['OQMD_L12']}")
print(f"  VASP: B2={counts['VASP_B2']}, L12={counts['VASP_L12']}, total={counts['VASP_B2']+counts['VASP_L12']}")
total = sum(counts.values())
total_b2 = counts['MP_B2'] + counts['OQMD_B2'] + counts['VASP_B2']
total_l12 = counts['MP_L12'] + counts['OQMD_L12'] + counts['VASP_L12']
print(f"  TOTAL: B2={total_b2}, L12={total_l12}, grand={total}")

# =====================================================================
# 2. Ω_sf coverage
# =====================================================================
print("\n[2] Loading compound data and computing Ω_sf...")
compound_df = load_compound_data()
print(f"  Total compounds loaded: {len(compound_df)}")

omega_sf = compute_dft_omega_sf(compound_df)
omega_b2, omega_l12 = compute_structure_specific_omega_sf(compound_df)

n_target = len(TARGET_ELEMENTS)
n_possible = n_target * (n_target - 1) // 2
print(f"\n  Target elements: {n_target}")
print(f"  Possible hetero pairs: {n_possible}")
print(f"  Combined Ω_sf (>=2 filter): {len(omega_sf)}")
print(f"  B2 Ω_sf (>=2 filter):      {len(omega_b2)}")
print(f"  L12 Ω_sf (>=2 filter):     {len(omega_l12)}")
print(f"  B2 coverage:  {len(omega_b2)}/{n_possible} = {len(omega_b2)/n_possible*100:.1f}%")
print(f"  L12 coverage: {len(omega_l12)}/{n_possible} = {len(omega_l12)/n_possible*100:.1f}%")

# Check HEA pair coverage
bcc_heas = [h for h in ALONSO_TABLE2 if h["struct"] == "BCC"]
fcc_heas = [h for h in ALONSO_TABLE2 if h["struct"] == "FCC"]

bcc_pairs_needed = set()
fcc_pairs_needed = set()
for h in bcc_heas:
    for a, b in combinations(sorted(h["comp"].keys()), 2):
        bcc_pairs_needed.add((a, b))
for h in fcc_heas:
    for a, b in combinations(sorted(h["comp"].keys()), 2):
        fcc_pairs_needed.add((a, b))

bcc_covered = sum(1 for p in bcc_pairs_needed if p in omega_b2)
fcc_covered = sum(1 for p in fcc_pairs_needed if p in omega_l12)
print(f"\n  BCC HEA pairs: {bcc_covered}/{len(bcc_pairs_needed)}")
print(f"  FCC HEA pairs: {fcc_covered}/{len(fcc_pairs_needed)}")

# Also check independent test pairs
bcc_ind_pairs = set()
fcc_ind_pairs = set()
for h in INDEPENDENT_TEST:
    for a, b in combinations(sorted(h["comp"].keys()), 2):
        if h["struct"] == "BCC":
            bcc_ind_pairs.add((a, b))
        else:
            fcc_ind_pairs.add((a, b))
bcc_ind_covered = sum(1 for p in bcc_ind_pairs if p in omega_b2)
fcc_ind_covered = sum(1 for p in fcc_ind_pairs if p in omega_l12)
print(f"  BCC Indep pairs: {bcc_ind_covered}/{len(bcc_ind_pairs)}")
print(f"  FCC Indep pairs: {fcc_ind_covered}/{len(fcc_ind_pairs)}")

# =====================================================================
# 3. Training set: γ/q optimization
# =====================================================================
print("\n[3] Training set γ optimization (64 HEAs)...")

N = len(ALONSO_TABLE2)
y_hea = np.array([h["a_exp"] for h in ALONSO_TABLE2])
structs = np.array([h["struct"] for h in ALONSO_TABLE2])
bcc = structs == "BCC"
fcc = structs == "FCC"
bcc_idx = np.where(bcc)[0]
fcc_idx = np.where(fcc)[0]

# Grid search for optimal γ
best_rmse_ss = 999
best_gb, best_gf = 0, 0

for gb in np.arange(-0.5, 2.51, 0.05):
    for gf in np.arange(-0.5, 2.51, 0.05):
        a_pred = np.zeros(N)
        for i in bcc_idx:
            a_pred[i] = compute_eq10_scaled(
                ALONSO_TABLE2[i]["comp"], "BCC", omega_b2, gb)
        for i in fcc_idx:
            a_pred[i] = compute_eq10_scaled(
                ALONSO_TABLE2[i]["comp"], "FCC", omega_l12, gf)
        r = np.sqrt(np.mean((y_hea - a_pred)**2))
        if r < best_rmse_ss:
            best_rmse_ss = r
            best_gb, best_gf = gb, gf

# Fine-tune
for gb in np.arange(best_gb - 0.05, best_gb + 0.06, 0.01):
    for gf in np.arange(best_gf - 0.05, best_gf + 0.06, 0.01):
        a_pred = np.zeros(N)
        for i in bcc_idx:
            a_pred[i] = compute_eq10_scaled(
                ALONSO_TABLE2[i]["comp"], "BCC", omega_b2, gb)
        for i in fcc_idx:
            a_pred[i] = compute_eq10_scaled(
                ALONSO_TABLE2[i]["comp"], "FCC", omega_l12, gf)
        r = np.sqrt(np.mean((y_hea - a_pred)**2))
        if r < best_rmse_ss:
            best_rmse_ss = r
            best_gb, best_gf = gb, gf

# Compute final predictions
a_eq10_ss = np.zeros(N)
for i in bcc_idx:
    a_eq10_ss[i] = compute_eq10_scaled(
        ALONSO_TABLE2[i]["comp"], "BCC", omega_b2, best_gb)
for i in fcc_idx:
    a_eq10_ss[i] = compute_eq10_scaled(
        ALONSO_TABLE2[i]["comp"], "FCC", omega_l12, best_gf)

# Vegard baseline
a_vegard = np.array([compute_vegard(h["comp"], h["struct"]) for h in ALONSO_TABLE2])

err_veg = y_hea - a_vegard
err_ss = y_hea - a_eq10_ss

print("\n  === Training Set Results (64 HEA) ===")
print(f"  q_BCC = {best_gb:.2f}, q_FCC = {best_gf:.2f}")
print(f"  Vegard  RMSE = {rmse(err_veg):.4f} Å")
print(f"  DFT-SS  RMSE = {rmse(err_ss):.4f} Å")
print(f"  Improvement:  {(1-rmse(err_ss)/rmse(err_veg))*100:.1f}%")
print(f"  BCC Vegard:  {rmse(err_veg[bcc]):.4f}, DFT-SS: {rmse(err_ss[bcc]):.4f}")
print(f"  FCC Vegard:  {rmse(err_veg[fcc]):.4f}, DFT-SS: {rmse(err_ss[fcc]):.4f}")

# =====================================================================
# 4. Independent test (20 HEA) — CRITICAL
# =====================================================================
print("\n[4] Independent Test Set (20 HEAs)...")

ind_results = []
for hea in INDEPENDENT_TEST:
    comp = hea["comp"]
    struct = hea["struct"]
    a_exp = hea["a_exp"]
    ref = hea.get("ref", "")

    a_veg = compute_vegard(comp, struct)
    omega_ss = omega_b2 if struct == "BCC" else omega_l12
    gamma_ss = best_gb if struct == "BCC" else best_gf
    a_ss = compute_eq10_scaled(comp, struct, omega_ss, gamma=gamma_ss)

    # Check how many pairs are covered
    elems = sorted(comp.keys())
    n_pairs = len(list(combinations(elems, 2)))
    n_covered = sum(1 for a, b in combinations(elems, 2)
                    if tuple(sorted([a, b])) in omega_ss)

    ind_results.append({
        "composition": "-".join(elems),
        "struct": struct,
        "a_exp": a_exp,
        "a_vegard": a_veg,
        "a_ss": a_ss,
        "err_vegard": a_exp - a_veg,
        "err_ss": a_exp - a_ss,
        "pairs_covered": f"{n_covered}/{n_pairs}",
        "vegard_degeneracy": abs(a_veg - a_ss) < 0.0001,
        "ref": ref,
    })

ind_df = pd.DataFrame(ind_results)
ind_df.to_csv(OUTDIR / "independent_test_results.csv", index=False)

bcc_ind = ind_df["struct"] == "BCC"
fcc_ind = ind_df["struct"] == "FCC"

err_veg_ind = ind_df["err_vegard"].values
err_ss_ind = ind_df["err_ss"].values

print("\n  === Independent Test Results ===")
print(f"  Vegard  RMSE = {rmse(err_veg_ind):.4f} Å")
print(f"  DFT-SS  RMSE = {rmse(err_ss_ind):.4f} Å")
print(f"  BCC ({bcc_ind.sum()}): Vegard={rmse(err_veg_ind[bcc_ind]):.4f}, DFT-SS={rmse(err_ss_ind[bcc_ind]):.4f}")
print(f"  FCC ({fcc_ind.sum()}): Vegard={rmse(err_veg_ind[fcc_ind]):.4f}, DFT-SS={rmse(err_ss_ind[fcc_ind]):.4f}")

# Check Vegard degeneracy
n_degen = ind_df[bcc_ind]["vegard_degeneracy"].sum()
print(f"\n  BCC Vegard degeneracy: {n_degen}/{bcc_ind.sum()} alloys")
print("  (Old paper claimed 7/8 — now should be much less)")

print("\n  Per-alloy detail:")
print(f"  {'Alloy':<30} {'St':>3} {'a_exp':>7} {'a_Veg':>7} {'a_SS':>7} {'Err_SS':>7} {'Cov':>5} {'Deg':>4}")
for _, r in ind_df.iterrows():
    deg = "V" if r["vegard_degeneracy"] else ""
    print(f"  {r['composition']:<30} {r['struct']:>3} {r['a_exp']:>7.4f} "
          f"{r['a_vegard']:>7.4f} {r['a_ss']:>7.4f} {r['err_ss']:>7.4f} "
          f"{r['pairs_covered']:>5} {deg:>4}")

# =====================================================================
# 5. Additive decomposition check
# =====================================================================
print("\n[5] Additive decomposition statistics...")

# How many unique pairs have Ω_sf data
# B2 and L12 combined
all_pairs_b2 = set(omega_b2.keys())
all_pairs_l12 = set(omega_l12.keys())
all_pairs = all_pairs_b2 | all_pairs_l12

print(f"  B2 DFT pairs:  {len(all_pairs_b2)}")
print(f"  L12 DFT pairs: {len(all_pairs_l12)}")
print(f"  Union:         {len(all_pairs)}")
print(f"  Total B2+L12 = {len(all_pairs_b2) + len(all_pairs_l12)}")

# =====================================================================
# 6. "905 pairs" verification
# =====================================================================
print("\n[6] Checking '905 pairs' claim...")

# Count pairs with all 3 structures: A3B L12, B3A L12, AB B2
# Need to parse from compound_df
pair_b2_raw = defaultdict(int)
pair_l12_a3b = defaultdict(int)  # A is majority (count=3)
pair_l12_b3a = defaultdict(int)  # A is minority (count=1)

for _, row in compound_df.iterrows():
    elA = row.get("element_A", "")
    elB = row.get("element_B", "")
    a = row.get("lattice_constant", 0)
    stype = row.get("stype", "")
    if a <= 2 or a >= 8 or elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
        continue
    if elA == elB:
        continue
    pair = tuple(sorted([elA, elB]))

    if stype == "B2":
        pair_b2_raw[pair] += 1
    elif stype == "L12":
        cA = row.get("count_A", 3)
        if cA == 3:  # A3B
            pair_l12_a3b[(elA, elB)] += 1
        else:  # A1B3 → B3A
            pair_l12_b3a[(elB, elA)] += 1

# Pairs with all 3 structures
count_all3 = 0
for pair in pair_b2_raw:
    a, b = pair
    # Check A-majority L12 exists for this pair (A3B or equivalently B in B3A)
    has_a3b = pair_l12_a3b.get((a, b), 0) > 0 or pair_l12_b3a.get((a, b), 0) > 0
    # Check B-majority L12 exists for this pair (B3A or equivalently A in A3B)
    has_b3a = pair_l12_a3b.get((b, a), 0) > 0 or pair_l12_b3a.get((b, a), 0) > 0
    if has_a3b and has_b3a:
        count_all3 += 1

print(f"  Pairs with all 3 structures (A3B + B3A + B2): {count_all3}")
print("  Paper claims: 905")

# Also count without requiring both L12 directions
count_any_l12 = 0
for pair in pair_b2_raw:
    a, b = pair
    has_any_l12 = (pair_l12_a3b.get((a, b), 0) > 0 or
                   pair_l12_a3b.get((b, a), 0) > 0 or
                   pair_l12_b3a.get((a, b), 0) > 0 or
                   pair_l12_b3a.get((b, a), 0) > 0)
    if has_any_l12:
        count_any_l12 += 1
print(f"  Pairs with B2 + any L12: {count_any_l12}")

# =====================================================================
# 7. Noise floor
# =====================================================================
print("\n[7] Noise floor...")
sigma_noise, dup_info = estimate_noise_floor(ALONSO_TABLE2)
print(f"  σ_noise = {sigma_noise:.4f} Å")

# =====================================================================
# 8. Convergence filtering
# =====================================================================
print("\n[8] Convergence filtering...")
# Count entries that pass lattice constant filter (2-8 Å)
total_raw = len(compound_df)
valid = compound_df["lattice_constant"].between(2, 8, inclusive="neither")
# Also need King volumes
has_king = (compound_df["element_A"].isin(KING_ATOMIC_VOLUMES) &
            compound_df["element_B"].isin(KING_ATOMIC_VOLUMES))
converged = (valid & has_king).sum()
print(f"  Raw entries: {total_raw}")
print(f"  After lattice filter (2<a<8) + King volumes: {converged}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY NUMBERS FOR MANUSCRIPT")
print("=" * 70)

summary = {
    "tab1_mp_b2": counts["MP_B2"],
    "tab1_mp_l12": counts["MP_L12"],
    "tab1_oqmd_b2": counts["OQMD_B2"],
    "tab1_oqmd_l12": counts["OQMD_L12"],
    "tab1_vasp_b2": counts["VASP_B2"],
    "tab1_vasp_l12": counts["VASP_L12"],
    "tab1_total": total,
    "tab1_total_b2": total_b2,
    "tab1_total_l12": total_l12,
    "omega_b2_pairs": len(omega_b2),
    "omega_l12_pairs": len(omega_l12),
    "omega_b2_pct": f"{len(omega_b2)/n_possible*100:.1f}",
    "omega_l12_pct": f"{len(omega_l12)/n_possible*100:.1f}",
    "bcc_hea_pairs": f"{bcc_covered}/{len(bcc_pairs_needed)}",
    "fcc_hea_pairs": f"{fcc_covered}/{len(fcc_pairs_needed)}",
    "q_bcc": best_gb,
    "q_fcc": best_gf,
    "train_rmse_vegard": f"{rmse(y_hea - a_vegard):.4f}",
    "train_rmse_ss": f"{rmse(err_ss):.4f}",
    "train_improvement": f"{(1-rmse(err_ss)/rmse(y_hea-a_vegard))*100:.1f}",
    "indep_rmse_vegard": f"{rmse(err_veg_ind):.4f}",
    "indep_rmse_ss": f"{rmse(err_ss_ind):.4f}",
    "indep_bcc_vegard": f"{rmse(err_veg_ind[bcc_ind]):.4f}",
    "indep_bcc_ss": f"{rmse(err_ss_ind[bcc_ind]):.4f}",
    "indep_fcc_vegard": f"{rmse(err_veg_ind[fcc_ind]):.4f}",
    "indep_fcc_ss": f"{rmse(err_ss_ind[fcc_ind]):.4f}",
    "indep_bcc_degeneracy": f"{n_degen}/{bcc_ind.sum()}",
    "sigma_noise": f"{sigma_noise:.4f}",
    "pairs_all3_structures": count_all3,
}

for k, v in summary.items():
    print(f"  {k}: {v}")

# Save
with open(OUTDIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to {OUTDIR}/")
print("Done.")
