#!/usr/bin/env python3
"""Experimental proof of structural vacancies on B2-NiAl from T&D density data.

Taylor & Doyle (1972, J. Appl. Cryst. 5, 201) measured both lattice parameters
and Archimedes densities for beta-NiAl alloys.  Table 2 gives the number of atoms
per unit cell n.  Because the conventional B2 cell contains two sites,

    c_vac = 1 - n / 2

is the vacancy fraction per lattice site on the Al-rich side.  This is an
independent experimental quantity: a(x) comes from X-ray diffraction, n comes from
density (mass/volume) and the same a.

On the Al-rich side the excess Al can in principle be accommodated by either
Ni vacancies or Al antisites.  This figure focuses on the vacancy concentration
per B2 lattice site:

    c_vac_model(x) = 1 - 1/(2x)          (structural Ni vacancies)
    c_vac_MLIP     = 1 - n_atoms/n_sites  (MACE-relaxed selected structure)
    c_vac^hybrid   = Boltzmann-weighted Ni-vacancy fraction

At x = 0.5 all curves coincide with perfect B2 (c_vac = 0).  The high-Al range
(x_Al > 0.62) is outside the stable single-phase B2 field and is shown as a
metastable extrapolation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, 'analysis')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 16, 'axes.grid': True, 'grid.alpha': 0.3,
                     'font.family': ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif'],
                     'axes.unicode_minus': False})

# --- Taylor & Doyle raw Table 2 ------------------------------------------------
td = pd.read_csv(os.path.join(AN, 'taylor_doyle_table2.csv'))
td_al = td[(td.x_Al_at > 50.0) & td.n_atoms_per_cell.notna()].sort_values('x_Al_at').copy()

# --- pure defect models ---------------------------------------------------------
def c_vac_model(x):
    return 1.0 - 1.0 / (2.0 * np.asarray(x, dtype=float))

# --- MACE selected-branch vacancy fraction -------------------------------------
# For each relaxed MACE structure the structural vacancy fraction is obtained
# directly from the lattice constant and the per-atom volume:
#     n_atoms/n_B2_cells = a^3 / V_atom   =>   c_vac = 1 - (a^3/V_atom)/2
# This is 0 for perfect B2 and for antisite structures (all sites occupied),
# and positive for vacancy structures.

def c_vac_mace_for_row(r):
    N = r.a_mix**3 / r.V_mix
    return float(np.clip(1.0 - N / 2.0, 0.0, 1.0))

mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
mask = (mix.x_Al >= 0.5) & (mix.x_Al <= 0.98)
mace = mix[mask].sort_values('x_Al').copy()
mace['c_vac_mace'] = mace.apply(c_vac_mace_for_row, axis=1)

# --- finite-temperature hybrid model -------------------------------------------
hyb = pd.read_csv(os.path.join(AN, 'b2_offstoich_hybrid_c_vac.csv'))
hyb = hyb[(hyb.x_Al >= 0.5) & (hyb.x_Al <= 0.98)].sort_values('x_Al')

x_grid = np.linspace(0.5, 0.98, 300)
c_mod = np.maximum(c_vac_model(x_grid), 0.0)

# Build a discontinuous MACE-selected c_vac line so we do not interpolate across
# the discrete vacancy -> antisite branch switch (e.g. x_Al ~0.9).
mace_sorted = mace.sort_values('x_Al').reset_index(drop=True)
x_mace_plot, c_mace_plot = [], []
prev_branch = None
for _, r in mace_sorted.iterrows():
    if prev_branch is not None and r.selected_branch != prev_branch:
        x_mace_plot.append(np.nan)
        c_mace_plot.append(np.nan)
    x_mace_plot.append(r.x_Al)
    c_mace_plot.append(r.c_vac_mace)
    prev_branch = r.selected_branch

# hybrid vacancy curves connect continuously to c=0 at x=0.5
c_hyb_1273 = np.interp(x_grid, hyb.x_Al.values, hyb.c_hybrid_1273K.values, left=0.0)
c_hyb_1473 = np.interp(x_grid, hyb.x_Al.values, hyb.c_hybrid_1473K.values, left=0.0)

# --- output table ---------------------------------------------------------------
table = []
for _, r in td_al.iterrows():
    x = r.x_Al_at / 100.0
    cmod = float(c_vac_model(x))
    a_m = float(np.interp(x, mace.x_Al.values, mace.a_mix.values))
    V_m = float(np.interp(x, mace.x_Al.values, mace.V_mix.values))
    N_m = a_m**3 / V_m
    c_m = float(np.clip(1.0 - N_m / 2.0, 0.0, 1.0))
    c_h1273 = float(np.interp(x, hyb.x_Al.values, hyb.c_hybrid_1273K.values, left=0.0))
    c_h1473 = float(np.interp(x, hyb.x_Al.values, hyb.c_hybrid_1473K.values, left=0.0))
    table.append({
        'x_Al': round(x, 4),
        'a_TD_A': round(r.a_A, 4),
        'rho_gcm3': r.rho_gcm3,
        'n_atoms_per_cell': round(r.n_atoms_per_cell, 3),
        'c_vac_exp': round(r.c_vac_exp, 4),
        'c_vac_model': round(cmod, 4),
        'c_vac_MLIP': round(c_m, 4),
        'c_vac_hybrid_1273K': round(c_h1273, 4),
        'c_vac_hybrid_1473K': round(c_h1473, 4),
    })
table = pd.DataFrame(table)
table.to_csv(os.path.join(AN, 'vacancy_concentration_exp_vs_mace.csv'), index=False)
print(table.to_string(index=False))

# --- plot -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=0.13, right=0.60, top=0.92, bottom=0.10)
ax.plot(x_grid, c_mod, 'k-', lw=2.5, label='$c_{\\rm vac}^{\\rm model}$ (Ni 空孔, $1-1/(2x)$)')
ax.plot(x_mace_plot, c_mace_plot, 'o--', color='tab:blue', lw=2, markersize=5,
        label='$c_{\\rm vac}^{\\rm MLIP}$ (MACE 選択構造)')
ax.plot(x_grid, c_hyb_1473, '-.', color='tab:orange', lw=2.5,
        label='$c_{\\rm vac}^{\\rm hybrid}$ (1473 K)')
ax.plot(x_grid, c_hyb_1273, ':', color='tab:purple', lw=2.5,
        label='$c_{\\rm vac}^{\\rm hybrid}$ (1273 K)')
ax.scatter(td_al.x_Al_at / 100.0, td_al.c_vac_exp, color='tab:red', s=80, zorder=5,
           label='$c_{\\rm vac}^{\\rm exp}$ (T&D 密度, Table 2)', edgecolors='k', linewidths=0.5)

# shade the metastable extrapolation region beyond the B2 single-phase field
ax.axvspan(0.65, 0.98, color='gray', alpha=0.08, zorder=0, label='_nolegend_')

ax.axhline(0.0, color='gray', lw=1.0, ls='--')
ax.axvline(0.5, color='gray', lw=1.0, ls=':')
ax.set_xlabel('$x_{\\rm Al}$', fontsize=18)
ax.set_ylabel('空孔率', fontsize=16)
ax.set_title('B2-NiAl Al 過剰側：空孔率', fontsize=18)
ax.set_xlim(0.50, 0.98)
ax.set_ylim(-0.05, 0.55)
ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.5, frameon=True)

# annotation: all curves converge to perfect B2 at x=0.5
ax.text(0.51, 0.47,
        '化学量論組成 $x_{\\rm Al}=0.50$ では完全 B2（$c_{\\rm vac}=0$）。\n'
        '有限温度では Boltzmann 混合した空孔率\n'
        '$c_{\\rm vac}^{\\rm hybrid}$ が T&D 実験と\n'
        'Ni 空孔モデルの間に位置する。\n'
        '$x_{\\rm Al}>0.65$ は B2 単相限界外の仮想的な外挿。',
        fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

out = os.path.join(FIG, 'fig_b2_vacancy_concentration.png')
plt.savefig(out, dpi=150)
plt.close()
print('Wrote', out)
