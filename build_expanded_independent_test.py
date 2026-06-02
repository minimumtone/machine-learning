#!/usr/bin/env python3
"""
Build an expanded, compositionally diverse independent test set.

Key improvements over the original 20-point test:
  - Hf-Nb-Ta-Ti-Zr reduced from 3 entries to 1 (eliminates BCC bias)
  - CoCrFeMnNi reduced from 4 entries to 1 (eliminates FCC bias)
  - CoCrFeNi reduced from 3 entries to 1
  - 5 new refractory BCC from Tseng 2019 (Hf-Mo-X subsets)
  - 5 noble metal FCC from Freudenberger 2017 (Au-Cu-Ni-Pd-Pt system)
  - CoFeMnNi (Cr-free) from Guo 2011 / Lucas 2012

Total: ~25 unique compositions with balanced BCC/FCC
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import csv
import pickle
from hea_lattice_xgboost import (
    KING_ATOMIC_VOLUMES, ALONSO_TABLE2, INDEPENDENT_TEST,
    compute_vegard, compute_eq10_scaled, compute_eq10_dft
)

# =====================================================================
# Load DFT Ω_sf from saved model bundle (pickle)
# =====================================================================
def load_model_bundle():
    """Load model bundle with omega_b2, omega_l12, gamma values."""
    pkl_path = os.path.join(os.path.dirname(__file__),
                            "hea_xgboost_output", "xgboost_model.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        print(f"Loaded model bundle from {pkl_path}")
        print(f"  omega_b2: {len(bundle.get('omega_b2', {}))} pairs")
        print(f"  omega_l12: {len(bundle.get('omega_l12', {}))} pairs")
        print(f"  gamma_bcc: {bundle.get('gamma_bcc', 'N/A')}")
        print(f"  gamma_fcc: {bundle.get('gamma_fcc', 'N/A')}")
        return bundle
    else:
        print(f"WARNING: {pkl_path} not found.")
        return None



# =====================================================================
# New expanded independent test dataset
# =====================================================================
NEW_INDEPENDENT_TEST = [
    # ===== BCC HEAs =====
    
    # --- Kept from original test (deduplicated, 1 per unique composition) ---
    # 1. MoNbTaV equiatomic (Yao 2016, Entropy 18, 189)
    {"comp": {"Mo":0.25, "Nb":0.25, "Ta":0.25, "V":0.25},
     "struct": "BCC", "a_exp": 3.208,
     "ref": "Yao2016_Entropy",
     "note": "equiatomic 4-element refractory"},
    
    # 2. AlNbTiV equiatomic (Stepanov 2015, J Alloys Compd 649, 130)
    {"comp": {"Al":0.25, "Nb":0.25, "Ti":0.25, "V":0.25},
     "struct": "BCC", "a_exp": 3.220,
     "ref": "Stepanov2015_JAlloyCompd",
     "note": "equiatomic 4-element with Al"},
    
    # 3. MoNbTaVW equiatomic (Kantelis 2025, AIP Adv; originally Senkov 2010)
    {"comp": {"Mo":0.20, "Nb":0.20, "Ta":0.20, "V":0.20, "W":0.20},
     "struct": "BCC", "a_exp": 3.185,
     "ref": "Kantelis2025_AIPAdv",
     "note": "equiatomic 5-element refractory"},
    
    # 4. HfNbTaTiZr equiatomic (Senkov 2012, Intermet 20, 87)
    #    Keep ONE representative value (3.404 from Youssef2015 = RT bulk)
    {"comp": {"Hf":0.20, "Nb":0.20, "Ta":0.20, "Ti":0.20, "Zr":0.20},
     "struct": "BCC", "a_exp": 3.404,
     "ref": "Senkov2012_Intermet",
     "note": "equiatomic 5-element Senkov standard"},
    
    # 5. Ti-rich HfNbTaTiZr variant (Dirras 2016, Mater Charact 111, 106)
    {"comp": {"Ti":0.35, "Zr":0.275, "Hf":0.275, "Nb":0.05, "Ta":0.05},
     "struct": "BCC", "a_exp": 3.440,
     "ref": "Dirras2016_MaterCharact",
     "note": "Ti-rich non-equiatomic 5-element"},
    
    # --- NEW: Tseng 2019, Entropy 21, 15 (Table 2) ---
    # Yeh group subtraction method: equiatomic 6-element and 5-element subsets
    # All single-phase BCC, XRD with Nelson-Riley extrapolation
    
    # 6. HfMoNbTaTiZr equiatomic (6-element base alloy)
    {"comp": {"Hf":1/6, "Mo":1/6, "Nb":1/6, "Ta":1/6, "Ti":1/6, "Zr":1/6},
     "struct": "BCC", "a_exp": 3.345,
     "ref": "Tseng2019_Entropy",
     "note": "equiatomic 6-element Yeh group"},
    
    # 7. HfMoTaTiZr (without Nb)
    {"comp": {"Hf":0.20, "Mo":0.20, "Ta":0.20, "Ti":0.20, "Zr":0.20},
     "struct": "BCC", "a_exp": 3.364,
     "ref": "Tseng2019_Entropy",
     "note": "equiatomic 5-element (no Nb)"},
    
    # 8. HfMoNbTiZr (without Ta)
    {"comp": {"Hf":0.20, "Mo":0.20, "Nb":0.20, "Ti":0.20, "Zr":0.20},
     "struct": "BCC", "a_exp": 3.369,
     "ref": "Tseng2019_Entropy",
     "note": "equiatomic 5-element (no Ta)"},
    
    # 9. HfMoNbTaZr (without Ti)
    {"comp": {"Hf":0.20, "Mo":0.20, "Nb":0.20, "Ta":0.20, "Zr":0.20},
     "struct": "BCC", "a_exp": 3.347,
     "ref": "Tseng2019_Entropy",
     "note": "equiatomic 5-element (no Ti)"},
    
    # 10. HfMoNbTaTi (without Zr)
    {"comp": {"Hf":0.20, "Mo":0.20, "Nb":0.20, "Ta":0.20, "Ti":0.20},
     "struct": "BCC", "a_exp": 3.305,
     "ref": "Tseng2019_Entropy",
     "note": "equiatomic 5-element (no Zr)"},
    
    # ===== FCC HEAs =====
    
    # --- Kept from original test (deduplicated) ---
    # 11. CoCrFeMnNi equiatomic Cantor alloy (Otto 2013, Acta Mater 61, 5743)
    {"comp": {"Co":0.20, "Cr":0.20, "Fe":0.20, "Mn":0.20, "Ni":0.20},
     "struct": "FCC", "a_exp": 3.5988,
     "ref": "Otto2013_ActaMat",
     "note": "Cantor alloy standard"},
    
    # 12. CoCrFeNi equiatomic (Wang 2019, Scripta Mater 162, 235)
    {"comp": {"Co":0.25, "Cr":0.25, "Fe":0.25, "Ni":0.25},
     "struct": "FCC", "a_exp": 3.5723,
     "ref": "Wang2019_ScriptaMat",
     "note": "equiatomic 4-element precision XRD"},
    
    # 13. Co0.5CrFeNi non-equiatomic (Wang 2019, Scripta Mater)
    {"comp": {"Co":0.143, "Cr":0.286, "Fe":0.286, "Ni":0.286},
     "struct": "FCC", "a_exp": 3.5805,
     "ref": "Wang2019_ScriptaMat",
     "note": "Co-lean non-equiatomic"},
    
    # 14. (CoCrFeNi)0.89Pd0.11 (Niu 2017, Sci Rep 7, 39803)
    {"comp": {"Co":0.2225, "Cr":0.2225, "Fe":0.2225, "Ni":0.2225, "Pd":0.11},
     "struct": "FCC", "a_exp": 3.620,
     "ref": "Niu2017_SciRep",
     "note": "11% Pd addition to CoCrFeNi"},
    
    # 15. CoCrFeNiPd equiatomic (Niu 2017, Sci Rep)
    {"comp": {"Co":0.20, "Cr":0.20, "Fe":0.20, "Ni":0.20, "Pd":0.20},
     "struct": "FCC", "a_exp": 3.660,
     "ref": "Niu2017_SciRep",
     "note": "equiatomic 5-element with Pd"},
    
    # 16. (CoCrFeNi)0.73Pd0.27 (Niu 2017, Sci Rep)
    {"comp": {"Co":0.1825, "Cr":0.1825, "Fe":0.1825, "Ni":0.1825, "Pd":0.27},
     "struct": "FCC", "a_exp": 3.710,
     "ref": "Niu2017_SciRep",
     "note": "Pd-rich 27% addition"},
    
    # 17. CoCrFeNiV equiatomic (Niu 2017, Sci Rep)
    {"comp": {"Co":0.20, "Cr":0.20, "Fe":0.20, "Ni":0.20, "V":0.20},
     "struct": "FCC", "a_exp": 3.610,
     "ref": "Niu2017_SciRep",
     "note": "equiatomic 5-element with V"},
    
    # --- NEW: Noble metal FCC HEAs ---
    # Freudenberger 2017, Metals 7, 135 (Table A1)
    # All single-phase FCC, Rietveld-refined lattice parameters
    
    # 18. AuCuNiPd equiatomic
    {"comp": {"Au":0.25, "Cu":0.25, "Ni":0.25, "Pd":0.25},
     "struct": "FCC", "a_exp": 3.8093,
     "ref": "Freudenberger2017_Metals",
     "note": "noble metal 4-element"},
    
    # 19. AuCuNiPt equiatomic
    {"comp": {"Au":0.25, "Cu":0.25, "Ni":0.25, "Pt":0.25},
     "struct": "FCC", "a_exp": 3.8107,
     "ref": "Freudenberger2017_Metals",
     "note": "noble metal 4-element with Pt"},
    
    # 20. AuCuPdPt equiatomic
    {"comp": {"Au":0.25, "Cu":0.25, "Pd":0.25, "Pt":0.25},
     "struct": "FCC", "a_exp": 3.8847,
     "ref": "Freudenberger2017_Metals",
     "note": "noble metal 4-element (no Ni)"},
    
    # 21. AuNiPdPt equiatomic
    {"comp": {"Au":0.25, "Ni":0.25, "Pd":0.25, "Pt":0.25},
     "struct": "FCC", "a_exp": 3.8738,
     "ref": "Freudenberger2017_Metals",
     "note": "noble metal 4-element (no Cu)"},
    
    # 22. CuNiPdPt equiatomic
    {"comp": {"Cu":0.25, "Ni":0.25, "Pd":0.25, "Pt":0.25},
     "struct": "FCC", "a_exp": 3.7622,
     "ref": "Freudenberger2017_Metals",
     "note": "noble metal 4-element (no Au)"},
    
    # 23. AuCuNiPdPt equiatomic (quinary)
    {"comp": {"Au":0.20, "Cu":0.20, "Ni":0.20, "Pd":0.20, "Pt":0.20},
     "struct": "FCC", "a_exp": 3.8307,
     "ref": "Freudenberger2017_Metals",
     "note": "noble metal 5-element quinary"},
    
]


def composition_label(comp):
    """Generate sorted composition label."""
    elements = sorted(comp.keys())
    parts = []
    for e in elements:
        c = comp[e]
        if abs(c - 1/len(elements)) < 0.01:
            parts.append(e)
        else:
            parts.append(f"{e}{c:.3f}")
    return "-".join(parts)


def main():
    bundle = load_model_bundle()
    if bundle is None:
        print("ERROR: Cannot proceed without model bundle.")
        return
    
    omega_b2 = bundle["omega_b2"]
    omega_l12 = bundle["omega_l12"]
    omega_sf_combined = bundle["omega_sf"]  # combined B2+L12 DFT Ω_sf
    gamma_bcc = bundle["gamma_bcc"]
    gamma_fcc = bundle["gamma_fcc"]
    
    print(f"\n{'='*80}")
    print(f"EXPANDED INDEPENDENT TEST SET: {len(NEW_INDEPENDENT_TEST)} HEAs")
    print(f"{'='*80}")
    
    bcc = [h for h in NEW_INDEPENDENT_TEST if h["struct"] == "BCC"]
    fcc = [h for h in NEW_INDEPENDENT_TEST if h["struct"] == "FCC"]
    print(f"  BCC: {len(bcc)}, FCC: {len(fcc)}")
    
    # Check unique element combinations
    def elem_key(comp):
        return tuple(sorted(comp.keys()))
    
    bcc_combos = set(elem_key(h["comp"]) for h in bcc)
    fcc_combos = set(elem_key(h["comp"]) for h in fcc)
    print(f"  Unique BCC element combos: {len(bcc_combos)}")
    print(f"  Unique FCC element combos: {len(fcc_combos)}")
    
    # Check overlap with training set
    training_combos = set()
    for h in ALONSO_TABLE2:
        key = (elem_key(h["comp"]), h["struct"])
        training_combos.add(key)
    
    print(f"\n  Training set has {len(training_combos)} unique (element-combo, struct) pairs")
    
    new_count = 0
    for h in NEW_INDEPENDENT_TEST:
        key = (elem_key(h["comp"]), h["struct"])
        is_new = key not in training_combos
        if is_new:
            new_count += 1
    print(f"  Test entries with element combos NOT in training: {new_count}/{len(NEW_INDEPENDENT_TEST)}")
    
    # Compute predictions
    results = []
    for h in NEW_INDEPENDENT_TEST:
        comp = h["comp"]
        struct = h["struct"]
        a_exp = h["a_exp"]
        
        a_vegard = compute_vegard(comp, struct)
        # Alonso Eq.10 with combined DFT Ω_sf, γ=1 (same as original "king" column)
        a_king = compute_eq10_dft(comp, struct, omega_sf_combined)
        # DFT-Ω_sf: structure-specific omega + optimized gamma
        if struct == "BCC":
            a_dft = compute_eq10_scaled(comp, struct, omega_b2, gamma_bcc)
        else:
            a_dft = compute_eq10_scaled(comp, struct, omega_l12, gamma_fcc)
        
        err_vegard = a_vegard - a_exp
        err_king = a_king - a_exp
        err_dft = a_dft - a_exp
        
        label = composition_label(comp)
        
        results.append({
            "composition": label,
            "struct": struct,
            "a_exp": a_exp,
            "a_vegard": a_vegard,
            "a_eq10_king": a_king,
            "a_eq10_ss": a_dft,
            "err_vegard": err_vegard,
            "err_king": err_king,
            "err_ss": err_dft,
            "ref": h["ref"],
            "note": h.get("note", ""),
        })
    
    # Write CSV
    outdir = os.path.join(os.path.dirname(__file__), "hea_xgboost_output")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "independent_test_results.csv")
    
    fieldnames = ["composition", "struct", "a_exp", "a_vegard", "a_eq10_king",
                   "a_eq10_ss", "err_vegard", "err_king", "err_ss", "ref", "note"]
    
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"\nWrote {len(results)} rows to {outpath}")
    
    # Print RMSE comparison
    print(f"\n{'='*80}")
    print("RMSE COMPARISON (ALL {})".format(len(results)))
    print(f"{'='*80}")
    
    errs_vegard = np.array([r["err_vegard"] for r in results])
    errs_king = np.array([r["err_king"] for r in results])
    errs_dft = np.array([r["err_ss"] for r in results])
    
    rmse_vegard = np.sqrt(np.mean(errs_vegard**2))
    rmse_king = np.sqrt(np.mean(errs_king**2))
    rmse_dft = np.sqrt(np.mean(errs_dft**2))
    
    mae_vegard = np.mean(np.abs(errs_vegard))
    mae_king = np.mean(np.abs(errs_king))
    mae_dft = np.mean(np.abs(errs_dft))
    
    print(f"  {'Method':<40s} {'RMSE (Å)':>10s} {'MAE (Å)':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Vegard (単純平均)':<40s} {rmse_vegard:>10.4f} {mae_vegard:>10.4f}")
    print(f"  {'Alonso 体積ズレ補正 (King実験値)':<40s} {rmse_king:>10.4f} {mae_king:>10.4f}")
    print(f"  {'本手法 DFT-Ωsf':<40s} {rmse_dft:>10.4f} {mae_dft:>10.4f}")
    
    # BCC/FCC breakdown
    for struct_name, struct_code in [("BCC", "BCC"), ("FCC", "FCC")]:
        subset = [r for r in results if r["struct"] == struct_code]
        if not subset:
            continue
        ev = np.array([r["err_vegard"] for r in subset])
        ek = np.array([r["err_king"] for r in subset])
        ed = np.array([r["err_ss"] for r in subset])
        print(f"\n  {struct_name} ({len(subset)} alloys):")
        print(f"    Vegard:            RMSE={np.sqrt(np.mean(ev**2)):.4f}, MAE={np.mean(np.abs(ev)):.4f}")
        print(f"    Alonso King:       RMSE={np.sqrt(np.mean(ek**2)):.4f}, MAE={np.mean(np.abs(ek)):.4f}")
        print(f"    DFT-Ωsf:          RMSE={np.sqrt(np.mean(ed**2)):.4f}, MAE={np.mean(np.abs(ed)):.4f}")
    
    # Individual predictions
    print(f"\n{'='*80}")
    print("INDIVIDUAL PREDICTIONS")
    print(f"{'='*80}")
    print(f"{'#':>3s} {'Composition':<28s} {'Str':>3s} {'a_exp':>7s} {'a_veg':>7s} {'a_king':>7s} {'a_dft':>7s} {'Δveg':>7s} {'Δdft':>7s}")
    print(f"{'':>3s} {'':28s} {'':>3s} {'(Å)':>7s} {'(Å)':>7s} {'(Å)':>7s} {'(Å)':>7s} {'(Å)':>7s} {'(Å)':>7s}")
    print("-" * 100)
    
    for i, r in enumerate(results):
        label = r["composition"][:27]
        print(f"{i+1:3d} {label:<28s} {r['struct']:>3s} "
              f"{r['a_exp']:7.4f} {r['a_vegard']:7.4f} {r['a_eq10_king']:7.4f} {r['a_eq10_ss']:7.4f} "
              f"{r['err_vegard']:+7.4f} {r['err_ss']:+7.4f}")
    
    print(f"\n{'='*80}")
    print("DIVERSITY ANALYSIS")
    print(f"{'='*80}")
    
    all_elements = set()
    for h in NEW_INDEPENDENT_TEST:
        all_elements.update(h["comp"].keys())
    print(f"  Total unique elements: {len(all_elements)}")
    print(f"  Elements: {', '.join(sorted(all_elements))}")
    
    n_elem_dist = {}
    for h in NEW_INDEPENDENT_TEST:
        n = len(h["comp"])
        n_elem_dist[n] = n_elem_dist.get(n, 0) + 1
    print(f"  Element count distribution: {dict(sorted(n_elem_dist.items()))}")
    
    unique_refs = set(h["ref"] for h in NEW_INDEPENDENT_TEST)
    print(f"  Unique references: {len(unique_refs)}")


if __name__ == "__main__":
    main()
