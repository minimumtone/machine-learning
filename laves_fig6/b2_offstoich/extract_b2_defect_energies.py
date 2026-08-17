#!/usr/bin/env python3
"""Extract composition-dependent vacancy and antisite formation energies for B2-NiAl.

Uses MACE-relaxed supercell energies and elemental chemical potentials (fcc Ni/Al).
Outputs `analysis/b2_defect_energies.csv`.

Definitions (per defect, relative to perfect B2):
  - Ni -> Al antisite (Al-rich antisite):  ΔE = E(Ni_{n-1}Al_{n+1}) - E(perfect) + μ_Ni - μ_Al
  - Al -> Ni antisite (Ni-rich antisite):  ΔE = E(Ni_{n+1}Al_{n-1}) - E(perfect) - μ_Ni + μ_Al
  - Ni vacancy (Al-rich, Ni sublattice):   ΔE = E(Ni_{n-1}Al_n)     - E(perfect) + μ_Ni
  - Al vacancy (Ni-rich, Al sublattice):   ΔE = E(Ni_nAl_{n-1})     - E(perfect) + μ_Al

Here μ_i are the fcc elemental reference energies (MACE-relaxed).
"""
import os, glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

mu = pd.read_csv(os.path.join(AN, "mace_mp_ref_results.csv"))
mu_Ni = float(mu[mu.label == "Ni"].energy_per_atom_eV.values[0])
mu_Al = float(mu[mu.label == "Al"].energy_per_atom_eV.values[0])

b2_files = glob.glob(os.path.join(AN, "b2_offstoich_volumes*.csv"))
df = pd.concat([pd.read_csv(f) for f in b2_files if os.path.exists(f)], ignore_index=True)
perfect = df[df.branch == "perfect"].iloc[0]
E_perfect = perfect.energy_eV
n_Ni_perfect = perfect.n_Ni
n_Al_perfect = perfect.n_Al
n_sites = perfect.n_sites  # 128

rows = []
for _, r in df.iterrows():
    if r.branch == "perfect":
        continue
    if not r.converged:
        continue
    dNi = r.n_Ni - n_Ni_perfect
    dAl = r.n_Al - n_Al_perfect
    dE = r.energy_eV - E_perfect
    # classify by branch and actual atom changes
    if r.branch == "antisite":
        if dNi > 0 and dAl < 0:
            # Ni-rich: extra Ni on Al sublattice, Al missing (Ni antisite)
            n_defect = abs(dNi)
            delta = (dE - dNi * mu_Ni - dAl * mu_Al) / n_defect
            kind = "Ni_antisite_on_Al"
        elif dNi < 0 and dAl > 0:
            # Al-rich: extra Al on Ni sublattice
            n_defect = abs(dAl)
            delta = (dE - dNi * mu_Ni - dAl * mu_Al) / n_defect
            kind = "Al_antisite_on_Ni"
        else:
            continue
    elif r.branch == "vacancy":
        if dNi < 0 and dAl == 0:
            n_defect = abs(dNi)
            delta = (dE + abs(dNi) * mu_Ni) / n_defect  # dE - dNi*mu_Ni because dNi negative
            kind = "Ni_vacancy"
        elif dAl < 0 and dNi == 0:
            n_defect = abs(dAl)
            delta = (dE + abs(dAl) * mu_Al) / n_defect
            kind = "Al_vacancy"
        else:
            continue
    else:
        continue

    rows.append({
        "x_Al_target": r.x_Al_target,
        "x_Al": r.x_Al,
        "branch": r.branch,
        "seed": r.seed,
        "defect_kind": kind,
        "n_defect": int(n_defect),
        "E_perfect_eV": E_perfect,
        "E_defect_eV": r.energy_eV,
        "deltaE_per_defect_eV": float(delta),
        "dE_total_eV": float(dE),
        "dNi": int(dNi),
        "dAl": int(dAl),
        "n_atoms": int(r.n_atoms),
        "n_sites": int(n_sites),
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(AN, "b2_defect_energies.csv"), index=False)
print(f"Wrote {len(out)} defect entries to {os.path.join(AN, 'b2_defect_energies.csv')}")
print(out.groupby(["defect_kind"])[["deltaE_per_defect_eV"]].agg(["mean", "std", "min", "max"]))
