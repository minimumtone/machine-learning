#!/usr/bin/env python3
"""B2-NiAl off-stoichiometry pipeline (Fig. 6(a) B2 branch, Yamanouchi & Miura 2018).

Probabilistic reproduction of the off-stoichiometric B2 Ni-Al average atomic
volume using the MACE-MP-0 MLIP with multiple randomly sampled point-defect
configurations per composition:

  Ni-rich side (x_Al < 0.5):  (A) Ni antisites on the Al sublattice
                              (B) Al (structural) vacancies
  Al-rich side (x_Al > 0.5):  (A) Ni vacancies (triple-defect side)
                              (B) Al antisites on the Ni sublattice

For each composition and defect branch, several random defect placements
(seeded) are relaxed; means, standard deviations, formation energies and
Boltzmann branch weights are reported so that the composition dependence of
V-bar(x_Al) is obtained probabilistically rather than from a single
configuration.

Outputs (single pass):
  analysis/b2_offstoich_volumes.csv   per-configuration results
  analysis/b2_offstoich_summary.json  aggregated statistics
  figures/fig_b2_offstoich_vbar.png   V-bar vs x_Al (both branches + exp.)
  figures/fig_b2_offstoich_a.png      effective lattice constant vs x_Al
  figures/fig_b2_offstoich_eform.png  formation energies / branch selection
  relax/*.extxyz                      relaxed structures
"""
import json
import os
import time

import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS

from mace.calculators import mace_mp  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
RELAX = os.path.join(BASE, "relax")
for d in (AN, FIG, RELAX):
    os.makedirs(d, exist_ok=True)

FMAX = 0.02  # eV/A
REP = 4      # 4x4x4 B2 supercell, 64 cells / 128 sites
NCELL = REP ** 3
NSEEDS = 3
KT_EV = 8.617333262e-5 * 1273.0  # Boltzmann weight temperature (1273 K)

CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")


def relax(atoms, name):
    atoms.calc = CALC
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    t0 = time.time()
    opt.run(fmax=FMAX, steps=500)
    conv = bool(opt.converged())
    e = float(atoms.get_potential_energy())
    v = float(atoms.get_volume())
    print(f"[relax] {name}: N={len(atoms)} E={e:.4f} eV V/atom={v/len(atoms):.4f} "
          f"conv={conv} ({time.time()-t0:.1f}s)", flush=True)
    ase_write(os.path.join(RELAX, name + ".extxyz"), atoms)
    return e, v, conv


def make_b2(a=2.88):
    at = bulk("NiAl", crystalstructure="cesiumchloride", a=a, cubic=True)
    return at.repeat((REP, REP, REP))  # Ni on corner, Al on center sublattice


def sublattice_indices(atoms):
    ni = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Ni"]
    al = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Al"]
    return ni, al


# --- pure references / stoichiometric B2 -----------------------------------
mu = {}
for el, struc, a0 in (("Ni", "fcc", 3.52), ("Al", "fcc", 4.05)):
    at = bulk(el, struc, a=a0, cubic=True).repeat((2, 2, 2))
    e, v, conv = relax(at, f"pure_{el}_{struc}")
    mu[el] = e / len(at)

rows = []


def record(name, branch, x_al_target, seed, e, v, n_ni, n_al, conv):
    n = n_ni + n_al
    x_al = n_al / n
    a_eff = (v / NCELL) ** (1.0 / 3.0)
    e_form = (e - n_ni * mu["Ni"] - n_al * mu["Al"]) / n
    rows.append(dict(
        structure_id=name, branch=branch, x_Al_target=x_al_target,
        x_Al=round(x_al, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
        n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
        V_per_atom_A3=v / n, a_eff_A=a_eff, E_form_eV_atom=e_form,
        converged=conv,
    ))


b2 = make_b2()
e, v, conv = relax(b2.copy(), "b2_x0.500_perfect")
record("b2_x0.500_perfect", "perfect", 0.5, 0, e, v, NCELL, NCELL, conv)

# --- off-stoichiometric configurations --------------------------------------
X_TARGETS = [0.42, 0.44, 0.46, 0.48, 0.52, 0.54, 0.56, 0.58]

for x in X_TARGETS:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(1000 * seed + int(round(x * 1000)))
        base = make_b2()
        ni_idx, al_idx = sublattice_indices(base)

        if x < 0.5:
            # branch A: Ni antisites on Al sublattice (n_sites fixed = 128)
            n_al = int(round(2 * NCELL * x))
            n_anti = NCELL - n_al
            at = base.copy()
            for i in rng.choice(al_idx, size=n_anti, replace=False):
                at[i].symbol = "Ni"
            name = f"b2_x{x:.3f}_antisiteNi_s{seed}"
            e, v, conv = relax(at, name)
            record(name, "antisite", x, seed, e, v, NCELL + n_anti, n_al, conv)

            # branch B: Al vacancies (Ni count fixed = 64 cells)
            n_al_v = int(round(NCELL * x / (1.0 - x)))
            n_vac = NCELL - n_al_v
            at = base.copy()
            del at[[int(i) for i in rng.choice(al_idx, size=n_vac, replace=False)]]
            name = f"b2_x{x:.3f}_vacAl_s{seed}"
            e, v, conv = relax(at, name)
            record(name, "vacancy", x, seed, e, v, NCELL, n_al_v, conv)
        else:
            # branch A: Ni vacancies (Al count fixed)
            n_ni_v = int(round(NCELL * (1.0 - x) / x))
            n_vac = NCELL - n_ni_v
            at = base.copy()
            del at[[int(i) for i in rng.choice(ni_idx, size=n_vac, replace=False)]]
            name = f"b2_x{x:.3f}_vacNi_s{seed}"
            e, v, conv = relax(at, name)
            record(name, "vacancy", x, seed, e, v, n_ni_v, NCELL, conv)

            # branch B: Al antisites on Ni sublattice
            n_ni = int(round(2 * NCELL * (1.0 - x)))
            n_anti = NCELL - n_ni
            at = base.copy()
            for i in rng.choice(ni_idx, size=n_anti, replace=False):
                at[i].symbol = "Al"
            name = f"b2_x{x:.3f}_antisiteAl_s{seed}"
            e, v, conv = relax(at, name)
            record(name, "antisite", x, seed, e, v, n_ni, NCELL + n_anti, conv)

# --- persist -----------------------------------------------------------------
import pandas as pd  # noqa: E402

df = pd.DataFrame(rows)
df.to_csv(os.path.join(AN, "b2_offstoich_volumes.csv"), index=False)

summary = {"mu_Ni_eV": mu["Ni"], "mu_Al_eV": mu["Al"],
           "T_boltzmann_K": 1273.0, "supercell": f"{REP}x{REP}x{REP}",
           "compositions": {}}
for x, grp in df[df.branch != "perfect"].groupby("x_Al_target"):
    entry = {}
    for br, g in grp.groupby("branch"):
        entry[br] = dict(
            V_mean=float(g.V_per_atom_A3.mean()), V_std=float(g.V_per_atom_A3.std(ddof=0)),
            a_mean=float(g.a_eff_A.mean()), a_std=float(g.a_eff_A.std(ddof=0)),
            E_form_mean=float(g.E_form_eV_atom.mean()),
            n_configs=int(len(g)),
        )
    if "antisite" in entry and "vacancy" in entry:
        dEf = entry["vacancy"]["E_form_mean"] - entry["antisite"]["E_form_mean"]
        # per-atom Boltzmann weight of the vacancy branch vs antisite branch
        w_vac = 1.0 / (1.0 + np.exp(dEf * 128 / KT_EV))
        entry["dE_form_vac_minus_anti_eV_atom"] = float(dEf)
        entry["boltzmann_weight_vacancy_128atoms_1273K"] = float(w_vac)
        entry["preferred_branch"] = "vacancy" if dEf < 0 else "antisite"
        wv, wa = w_vac, 1.0 - w_vac
        entry["V_boltzmann_A3"] = float(wv * entry["vacancy"]["V_mean"] + wa * entry["antisite"]["V_mean"])
    summary["compositions"][f"{x:.3f}"] = entry

perfect = df[df.branch == "perfect"].iloc[0]
summary["V_B2_perfect_A3"] = float(perfect.V_per_atom_A3)
summary["a_B2_perfect_A"] = float(perfect.a_eff_A)

with open(os.path.join(AN, "b2_offstoich_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("DONE")
