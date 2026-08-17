#!/usr/bin/env python3
"""Fixed-cell relaxation test for Al-rich B2-NiAl.

For selected Al-rich configurations, fix the cell to the Taylor & Doyle
experimental a(x_Al) and relax only internal coordinates.  This directly
tests whether MACE over-relaxes the cell around Ni vacancies; if so, the
vacancy branch would rise relative to the antisite branch and x_max would
shift to lower x.
"""
import os
import time
import numpy as np
import pandas as pd
from ase.io import read
from ase.optimize import LBFGS
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))

CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")

FMAX = 0.02
T_ANNEAL_K = 1273.0
KT_EV = 8.617333262e-5 * T_ANNEAL_K


def a_taylor_doyle(x_Al):
    x_Ni = 1.0 - x_Al
    if x_Al <= 0.5:
        a = 2.8870 + (2.8618 - 2.8870) / (0.66 - 0.50) * (x_Ni - 0.50)
    else:
        a = 2.8870 + (2.8652 - 2.8870) / (0.45 - 0.50) * (x_Ni - 0.50)
    return a


def relax_fixed_cell(atoms, name):
    atoms.calc = CALC
    opt = LBFGS(atoms, logfile=None)
    t0 = time.time()
    opt.run(fmax=FMAX, steps=500)
    e = float(atoms.get_potential_energy())
    v = float(atoms.get_volume())
    print(f"[fix] {name}: E={e:.4f} eV V/atom={v/len(atoms):.4f} conv={opt.converged()} ({time.time()-t0:.1f}s)", flush=True)
    return e, v, opt.converged()


def get_mus():
    # Pure element reference energies from already relaxed structures, if present
    for el, fn in [("Ni", "pure_Ni_fcc.extxyz"), ("Al", "pure_Al_fcc.extxyz")]:
        path = os.path.join(BASE, "relax", fn)
        if os.path.exists(path):
            at = read(path)
            at.calc = CALC
            yield el, at.get_potential_energy() / len(at)

mu = dict(get_mus())
print("mu:", mu)


def run(x_Al, seed=0):
    a_fix = a_taylor_doyle(x_Al)
    results = []
    for branch, filename in [
        ("vacancy", f"b2_x0.{int(round(x_Al*1000)):03d}_vacNi_r4_s{seed}.extxyz"),
        ("antisite", f"b2_x0.{int(round(x_Al*1000)):03d}_antisiteAl_s{seed}.extxyz"),
    ]:
        path = os.path.join(BASE, "relax", filename)
        if not os.path.exists(path):
            # try alternative filename without _r4
            alt = filename.replace("_r4", "")
            path = os.path.join(BASE, "relax", alt)
        if not os.path.exists(path):
            print(f"Skip {filename} (not found)")
            continue
        at = read(path)
        # original fully-relaxed values
        at.calc = CALC
        e_free = at.get_potential_energy()
        v_free = at.get_volume()
        n = len(at)

        # fix cell to experimental a and relax internals only
        at_fix = at.copy()
        # The stored configurations are 4x4x4 B2 supercells (64 conventional cells)
        # so the supercell edge to match the experimental conventional a is 4*a_fix.
        at_fix.set_cell([4.0 * a_fix, 4.0 * a_fix, 4.0 * a_fix], scale_atoms=False)
        at_fix.set_pbc(True)
        e_fix, v_fix, conv = relax_fixed_cell(at_fix, f"{branch}_x{x_Al:.2f}_seed{seed}")

        rows = []
        for tag, energy, volume in [("free", e_free, v_free), ("fixed_cell", e_fix, v_fix)]:
            x_actual = at.get_atomic_numbers().tolist().count(13) / len(at) if tag == "free" else np.nan
            results.append({
                'x_Al_target': x_Al,
                'branch': branch,
                'seed': seed,
                'constr': tag,
                'a_cell_A': a_fix if tag == 'fixed_cell' else (v_free / 64.0) ** (1.0 / 3.0),
                'E_eV': energy,
                'V_atom_A3': volume / n,
                'n_atoms': n,
                'x_Al_actual': x_actual,
                'conv': tag == 'free' or conv,
            })
    return results


# Run at x=0.60 and, if available, x=0.58
all_rows = []
for x in [0.60, 0.58]:
    try:
        all_rows.extend(run(x, seed=0))
    except Exception as e:
        print(f"Error at x={x}: {e}")

df = pd.DataFrame(all_rows)
if not df.empty:
    df.to_csv(os.path.join(BASE, "analysis", "fixed_cell_relaxation_test.csv"), index=False)
    print(df.to_string(index=False))
else:
    print("No results")
