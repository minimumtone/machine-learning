#!/usr/bin/env python3
"""Finite-temperature hybrid vacancy/antisite model for B2-NiAl.

For each target composition and temperature, Boltzmann-weight the vacancy
and antisite branches using the per-atom Helmholtz free energy G_i:

    p_i = exp(-beta G_i) / sum_j exp(-beta G_j)
    c_vac^hybrid = sum_i p_i (1 - n_atoms_i / n_sites_i)

G_i is re-computed at the requested temperatures from the 0 K formation
energy Ef and the configurational degeneracy g_i = C(n_sub, n_defect) of the
point-defect sublattice, where n_sub = n_sites / 2.

Outputs:
  analysis/b2_offstoich_hybrid_c_vac.csv : hybrid c_vac and Al-antisite
                                            fractions at 1273 and 1473 K
"""
import os, math
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, 'analysis')
KB_EV = 8.617333262e-5
T_LIST = [1273.0, 1473.0]


def ln_comb(n, k):
    if k < 0 or k > n or n < 0:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def c_vac_from_n(n, n_sites):
    """Vacancy fraction per lattice site for a B2 cell with n_sites total sites."""
    return 1.0 - float(n) / n_sites


def n_sublattice(row):
    """Number of sites per B2 sublattice from the stored n_sites value."""
    return int(round(row.n_sites / 2.0))


def p_Al_antisite_from_c(c, x, n_sub):
    """Al-antisite fraction on the Ni sublattice for Al-rich hybrid c_vac.

    Fixed-composition constraint for Al-rich:
        n_Al = n_sub + n_a, n_Ni = n_sub - n_v - n_a,
        N = 2*n_sub - n_v, x = (n_sub + n_a) / N, n_v = 2*n_sub * c_vac.
    Returns NaN for x <= 0.5 (not meaningful as Al antisite variable).
    """
    if x <= 0.5:
        return np.nan
    n_v = 2 * n_sub * c
    N = 2 * n_sub - n_v
    n_a = x * N - n_sub
    return np.clip(n_a / float(n_sub), 0.0, 1.0)


def n_Ni_Al_from_row(row):
    """Recover integer Ni/Al counts from the mean composition and atom count."""
    n_at = int(round(row.n_atoms))
    n_Al = int(round(row.x_Al * n_at))
    n_Ni = n_at - n_Al
    return n_Ni, n_Al


def n_defect_from_row(row):
    """Number of point defects on the minority sublattice.

    Follows the same convention as make_figures.py / analyze_b2_hull_xmax.py
    but uses the per-row n_sites so it is valid for both 4x4x4 and 5x5x5 cells.
    """
    n_Ni, n_Al = n_Ni_Al_from_row(row)
    n_sub = n_sublattice(row)
    if row.branch == 'perfect':
        return 0
    if row.branch == 'antisite':
        if row.x_Al < 0.5:
            return max(n_Ni - n_sub, 0)   # Ni on Al sublattice
        else:
            return max(n_Al - n_sub, 0)   # Al on Ni sublattice
    # vacancy branch
    if row.x_Al < 0.5:
        return max(n_sub - n_Al, 0)       # vacancies on Al sublattice
    else:
        return max(n_sub - n_Ni, 0)       # vacancies on Ni sublattice


def perfect_G_eV():
    """Return the per-atom formation energy (G at T=0) for perfect B2."""
    vol_path = os.path.join(AN, 'b2_offstoich_volumes.csv')
    if os.path.exists(vol_path):
        vol = pd.read_csv(vol_path)
        p = vol[vol.branch == 'perfect']
        if not p.empty:
            return float(p.iloc[0].E_form_eV_atom)
    raise FileNotFoundError(
        "Perfect B2 formation energy not found; provide analysis/b2_offstoich_volumes.csv"
    )


def main():
    bm = pd.read_csv(os.path.join(AN, 'b2_offstoich_branch_means.csv'))
    perfect_Ef = perfect_G_eV()

    # Build the working branch table (no perfect row) and add per-row quantities.
    br = bm[bm.branch != 'perfect'].copy()
    br['n_defect'] = br.apply(n_defect_from_row, axis=1)
    br['c_vac'] = br.apply(lambda r: c_vac_from_n(r.n_atoms, r.n_sites), axis=1)

    # At x=0.50 both defect branches collapse to perfect B2.  Add a synthetic
    # perfect point to each branch so that the PCHIP interpolators pass through
    # (0.5, perfect_Ef) and the curves meet continuously at the ideal composition.
    # n_sites does not matter here because n_defect=0 (ln C=0).
    perfect_rows = pd.DataFrame([
        {'branch': 'vacancy', 'x_Al_target': 0.5, 'x_Al': 0.5,
         'Ef': perfect_Ef, 'n_atoms': 128, 'n_sites': 128,
         'n_defect': 0, 'c_vac': 0.0},
        {'branch': 'antisite', 'x_Al_target': 0.5, 'x_Al': 0.5,
         'Ef': perfect_Ef, 'n_atoms': 128, 'n_sites': 128,
         'n_defect': 0, 'c_vac': 0.0},
    ])
    br = pd.concat([br, perfect_rows], ignore_index=True)

    # Helmholtz free energy per atom for the two requested temperatures.
    for T in T_LIST:
        kT = KB_EV * T
        br[f'G_{T:.0f}'] = br.apply(
            lambda r: r.Ef - kT * ln_comb(n_sublattice(r), r.n_defect) / r.n_atoms,
            axis=1,
        )

    # Build monotone PCHIP interpolators per branch for G(T) and c_vac(x).
    splines = {}
    for br_name in ('vacancy', 'antisite'):
        sub = br[br.branch == br_name].sort_values('x_Al').copy()
        sub = sub.groupby('x_Al', as_index=False).agg({
            'c_vac': 'mean',
            **{f'G_{T:.0f}': 'mean' for T in T_LIST},
        })
        x = sub.x_Al.values
        splines[br_name] = {
            'x': x,
            'xmin': float(x.min()),
            'xmax': float(x.max()),
            'c': PchipInterpolator(x, sub['c_vac'].values, extrapolate=False),
        }
        for T in T_LIST:
            splines[br_name][T] = PchipInterpolator(
                x, sub[f'G_{T:.0f}'].values, extrapolate=False
            )

    # Dense output grid from stoichiometry up to the common branch limit.
    xmax = min(splines['vacancy']['xmax'], splines['antisite']['xmax'])
    x_grid = np.linspace(0.5, xmax, max(2, int(round((xmax - 0.5) / 0.0025)) + 1))

    rows = []
    for x in x_grid:
        c_mod = max(0.0, 1.0 - 1.0 / (2.0 * x))
        row = {'x_Al_target': round(x, 6), 'x_Al': round(x, 6), 'c_model': c_mod}
        for T in T_LIST:
            kT = KB_EV * T
            gvals = {}
            cvals = {}
            for br_name in ('vacancy', 'antisite'):
                s = splines[br_name]
                in_dom = (x + 1e-9 >= s['xmin']) and (x - 1e-9 <= s['xmax'])
                if in_dom:
                    g_val = float(s[T](x))
                    c_val = float(s['c'](x))
                    if not (np.isnan(g_val) or np.isnan(c_val)):
                        gvals[br_name] = g_val
                        cvals[br_name] = c_val
                        row[f'G_{br_name}_{T:.0f}K'] = g_val
                    else:
                        row[f'G_{br_name}_{T:.0f}K'] = np.nan
                else:
                    row[f'G_{br_name}_{T:.0f}K'] = np.nan

            if len(gvals) >= 2:
                gmin = min(gvals.values())
                weights = {
                    b: math.exp(-(g - gmin) / kT)
                    for b, g in gvals.items() if not np.isnan(g)
                }
                z = sum(weights.values())
                probs = {b: w / z for b, w in weights.items()}
                # antisite branch has c_vac=0, vacancy branch has the structural
                # vacancy fraction from the PCHIP c_vac(x) estimator.
                c_hybrid = sum(probs[b] * cvals.get(b, 0.0) for b in probs)
                if x > 0.5:
                    # n_sub cancels in p_Al_antisite_from_c; use 64 as a dummy.
                    p_anti = p_Al_antisite_from_c(c_hybrid, x, 64)
                    if pd.isna(p_anti):
                        p_anti = 0.0
                else:
                    p_anti = 0.0
            else:
                c_hybrid = np.nan
                p_anti = np.nan
                probs = {}

            c_total = c_hybrid + p_anti / 2.0 if not pd.isna(p_anti) else np.nan
            row[f'c_hybrid_{T:.0f}K'] = c_hybrid
            row[f'p_antisite_{T:.0f}K'] = p_anti
            row[f'c_total_{T:.0f}K'] = c_total
            for br_name in ('vacancy', 'antisite'):
                row[f'prob_{br_name}_{T:.0f}K'] = probs.get(br_name, np.nan)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('x_Al_target')
    out.to_csv(os.path.join(AN, 'b2_offstoich_hybrid_c_vac.csv'), index=False)
    print(out[['x_Al_target','c_model','c_hybrid_1273K','p_antisite_1273K','c_total_1273K','c_hybrid_1473K','p_antisite_1473K','c_total_1473K']].to_string(index=False))
    print('Wrote', os.path.join(AN, 'b2_offstoich_hybrid_c_vac.csv'))


if __name__ == '__main__':
    main()
