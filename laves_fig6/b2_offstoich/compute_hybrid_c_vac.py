#!/usr/bin/env python3
"""Finite-temperature hybrid vacancy/antisite model for B2-NiAl.

For each target composition and temperature, Boltzmann-weight the vacancy
and antisite branches using the per-atom Helmholtz free energy G_i:

    p_i = exp(-beta G_i) / sum_j exp(-beta G_j)
    c_vac^hybrid = sum_i p_i (1 - n_atoms_i / 128)

G_i is re-computed at the requested temperatures from the 0 K formation
energy Ef and the configurational degeneracy g_i = C(64, n_defect) of the
point-defect sublattice.

Outputs:
  analysis/b2_offstoich_hybrid_c_vac.csv : hybrid c_vac and Al-antisite
                                            fractions at 1273 and 1473 K
"""
import os, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, 'analysis')
NCELL = 64
N_SITES = 2 * NCELL
KB_EV = 8.617333262e-5
T_LIST = [1273.0, 1473.0]


def ln_comb(n, k):
    if k < 0 or k > n or n < 0:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def c_vac_from_n(n):
    """Vacancy fraction per lattice site for a 128-site B2 cell."""
    return 1.0 - float(n) / N_SITES


def p_Al_antisite_from_c(c, x):
    """Al-antisite fraction on the Ni sublattice for Al-rich hybrid c_vac.

    Fixed-composition constraint for Al-rich:
        n_Al = 64 + n_a, n_Ni = 64 - n_v - n_a,
        N = 128 - n_v, x = (64 + n_a) / N, n_v = 128 * c_vac.
    Returns NaN for x <= 0.5 (not meaningful as Al antisite variable).
    """
    if x <= 0.5:
        return np.nan
    n_v = N_SITES * c
    N = N_SITES - n_v
    n_a = x * N - NCELL
    return np.clip(n_a / float(NCELL), 0.0, 1.0)


def main():
    bm = pd.read_csv(os.path.join(AN, 'b2_offstoich_branch_means.csv'))
    br = bm[bm.branch != 'perfect'].copy()

    def n_defect_from_row(row):
        n = int(round(row.n_atoms))
        x = row.x_Al_target
        if row.branch == 'perfect':
            return 0
        if row.branch == 'antisite':
            if x < 0.5:
                # Ni on Al sublattice: n = 128 + n_anti
                return int(round(n - N_SITES))
            else:
                # Al on Ni sublattice: n = 128 (n_anti counted by excess Al)
                return int(round(n - N_SITES))
        else:  # vacancy
            return int(round(N_SITES - n))

    br['n_defect'] = br.apply(n_defect_from_row, axis=1)

    rows = []
    for xt, g in br.groupby('x_Al_target'):
        x_eff = float(g.x_Al.mean())
        c_mod = max(0.0, 1.0 - 1.0 / (2.0 * x_eff))
        row = {'x_Al_target': xt, 'x_Al': round(x_eff, 6), 'c_model': c_mod}
        branches = [b for b in ('vacancy', 'antisite') if b in g.branch.values]
        for T in T_LIST:
            kT = KB_EV * T
            gvals = {}
            cvals = {}
            for _, r in g.iterrows():
                g_T = r.Ef - KB_EV * T * ln_comb(NCELL, r.n_defect) / r.n_atoms
                gvals[r.branch] = g_T
                cvals[r.branch] = c_vac_from_n(r.n_atoms)
            if len(branches) >= 2:
                gmin = min(gvals.values())
                weights = {b: math.exp(-(g - gmin) / kT) for b, g in gvals.items()}
                z = sum(weights.values())
                probs = {b: w / z for b, w in weights.items()}
                c_hybrid = sum(probs[b] * cvals[b] for b in probs)
            elif len(branches) == 1:
                b0 = branches[0]
                c_hybrid = cvals[b0]
                probs = {b0: 1.0}
            else:
                c_hybrid = c_mod
                probs = {}
            # For Ni-rich side the relevant defect is Ni antisite on the Al
            # sublattice; the Al-antisite variable below is not meaningful.
            if x_eff <= 0.5:
                c_hybrid = 0.0
                p_anti = 0.0
            else:
                p_anti = p_Al_antisite_from_c(c_hybrid, x_eff)
                if pd.isna(p_anti):
                    p_anti = 0.0
            row[f'c_hybrid_{T:.0f}K'] = c_hybrid
            row[f'p_antisite_{T:.0f}K'] = p_anti
            for b in ('vacancy', 'antisite'):
                row[f'prob_{b}_{T:.0f}K'] = probs.get(b, np.nan)
                if b in gvals:
                    row[f'G_{b}_{T:.0f}K'] = gvals[b]
        rows.append(row)

    # Insert a perfect-B2 row at x=0.5 so the hybrid curve is continuous and
    # equals zero right at stoichiometry.
    perfect_row = {'x_Al_target': 0.5, 'x_Al': 0.5, 'c_model': 0.0}
    for T in T_LIST:
        perfect_row[f'c_hybrid_{T:.0f}K'] = 0.0
        perfect_row[f'p_antisite_{T:.0f}K'] = 0.0
        perfect_row[f'prob_vacancy_{T:.0f}K'] = 0.0
        perfect_row[f'prob_antisite_{T:.0f}K'] = 0.0
        perfect_row[f'G_vacancy_{T:.0f}K'] = -0.6920193068237968
        perfect_row[f'G_antisite_{T:.0f}K'] = -0.6920193068237968
    rows.append(perfect_row)

    out = pd.DataFrame(rows).sort_values('x_Al_target')
    out.to_csv(os.path.join(AN, 'b2_offstoich_hybrid_c_vac.csv'), index=False)
    print(out[['x_Al_target','c_model','c_hybrid_1273K','p_antisite_1273K','c_hybrid_1473K','p_antisite_1473K']].to_string(index=False))
    print('Wrote', os.path.join(AN, 'b2_offstoich_hybrid_c_vac.csv'))


if __name__ == '__main__':
    main()
