#!/usr/bin/env python3
"""Experimental proof of structural vacancies on B2-NiAl from T&D density data.

Taylor & Doyle (1972, J. Appl. Cryst. 5, 201) measured both lattice parameters
and Archimedes densities for beta-NiAl alloys.  Table 2 gives the number of atoms
per unit cell n.  Because the conventional B2 cell contains two sites,

    c_vac = 1 - n / 2

is the vacancy fraction per lattice site on the Al-rich side.  This is an
independent experimental quantity: a(x) comes from X-ray diffraction, n comes from
density (mass/volume) and the same a.

On the Al-rich side the excess Al can be accommodated by either
1. Ni vacancies:   c_vac_model(x) = 1 - 1/(2x)
2. Al antisites on Ni sites:  c_antisite_model(x) = x - 0.5 (per lattice site)
At x = 0.5 both models give zero defects and coincide with perfect B2.  At finite
temperature the two pure-defect branches are Boltzmann-mixed; the hybrid total
Ni-sublattice defect fraction is c_total = c_vac + p_antisite/2.
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

def c_antisite_model(x):
    # Al on Ni sublattice, per total B2 lattice site
    return np.maximum(np.asarray(x, dtype=float) - 0.5, 0.0)

# --- MACE selected branch -------------------------------------------------------
mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
mask = (mix.x_Al >= 0.45) & (mix.x_Al <= 0.66)
mace = mix[mask].sort_values('x_Al')

# --- finite-temperature hybrid model --------------------------------------------
hyb = pd.read_csv(os.path.join(AN, 'b2_offstoich_hybrid_c_vac.csv'))
hyb = hyb[(hyb.x_Al >= 0.45) & (hyb.x_Al <= 0.66)].sort_values('x_Al')

# restrict the hybrid model to the Al-rich side where the structural-vacancy
# interpretation (Ni-site vacancies) is meaningful
hyb = hyb[(hyb.c_hybrid_1473K.notna()) | (hyb.x_Al == 0.5)].copy()
hyb = hyb[(hyb.x_Al >= 0.5) & (hyb.x_Al <= 0.66)].sort_values('x_Al')

x_grid = np.linspace(0.45, 0.60, 200)
c_mod = np.maximum(c_vac_model(x_grid), 0.0)
c_anti = c_antisite_model(x_grid)

a_mace = np.interp(x_grid, mace.x_Al.values, mace.a_mix.values)
V_mace = np.interp(x_grid, mace.x_Al.values, mace.V_mix.values)
N_mace = a_mace**3 / V_mace
c_mace = 1.0 - N_mace / 2.0
c_mace = np.clip(c_mace, 0.0, 1.0)

# hybrid curves only for x_Al >= 0.5; they connect continuously to c=0 at x=0.5
c_hyb_1273 = np.full_like(x_grid, np.nan, dtype=float)
c_hyb_1473 = np.full_like(x_grid, np.nan, dtype=float)
c_total_1273 = np.full_like(x_grid, np.nan, dtype=float)
c_total_1473 = np.full_like(x_grid, np.nan, dtype=float)
mask = x_grid >= 0.5
c_hyb_1273[mask] = np.interp(x_grid[mask], hyb.x_Al.values, hyb.c_hybrid_1273K.values, left=0.0)
c_hyb_1473[mask] = np.interp(x_grid[mask], hyb.x_Al.values, hyb.c_hybrid_1473K.values, left=0.0)
c_total_1273[mask] = np.interp(x_grid[mask], hyb.x_Al.values, hyb.c_total_1273K.values, left=0.0)
c_total_1473[mask] = np.interp(x_grid[mask], hyb.x_Al.values, hyb.c_total_1473K.values, left=0.0)

# --- output table ---------------------------------------------------------------
table = []
for _, r in td_al.iterrows():
    x = r.x_Al_at / 100.0
    cmod = float(c_vac_model(x))
    canti = float(c_antisite_model(x))
    a_m = float(np.interp(x, mace.x_Al.values, mace.a_mix.values))
    V_m = float(np.interp(x, mace.x_Al.values, mace.V_mix.values))
    N_m = a_m**3 / V_m
    c_m = float(np.clip(1.0 - N_m / 2.0, 0.0, 1.0))
    c_h1273 = float(np.interp(x, hyb.x_Al.values, hyb.c_hybrid_1273K.values, left=0.0))
    c_h1473 = float(np.interp(x, hyb.x_Al.values, hyb.c_hybrid_1473K.values, left=0.0))
    c_t1273 = float(np.interp(x, hyb.x_Al.values, hyb.c_total_1273K.values, left=0.0))
    c_t1473 = float(np.interp(x, hyb.x_Al.values, hyb.c_total_1473K.values, left=0.0))
    p_a1273 = float(np.interp(x, hyb.x_Al.values, hyb.p_antisite_1273K.values, left=0.0))
    p_a1473 = float(np.interp(x, hyb.x_Al.values, hyb.p_antisite_1473K.values, left=0.0))
    table.append({
        'x_Al': round(x, 4),
        'a_TD_A': round(r.a_A, 4),
        'rho_gcm3': r.rho_gcm3,
        'n_atoms_per_cell': round(r.n_atoms_per_cell, 3),
        'c_vac_exp': round(r.c_vac_exp, 4),
        'c_vac_model': round(cmod, 4),
        'c_antisite_model': round(canti, 4),
        'c_vac_MLIP': round(c_m, 4),
        'c_vac_hybrid_1273K': round(c_h1273, 4),
        'c_vac_hybrid_1473K': round(c_h1473, 4),
        'c_total_hybrid_1273K': round(c_t1273, 4),
        'c_total_hybrid_1473K': round(c_t1473, 4),
        'p_Al_antisite_1273K': round(p_a1273, 4),
        'p_Al_antisite_1473K': round(p_a1473, 4),
    })
table = pd.DataFrame(table)
table.to_csv(os.path.join(AN, 'vacancy_concentration_exp_vs_mace.csv'), index=False)
print(table.to_string(index=False))

# --- plot -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=0.17, right=0.98, top=0.92, bottom=0.10)
ax.plot(x_grid, c_mod, 'k-', lw=2.5, label='$c_{\\rm vac}^{\\rm model}$ (Ni 空孔, $1-1/(2x)$)')
ax.plot(x_grid, c_anti, '--', color='tab:green', lw=2.0,
        label='$c_{\\rm anti}^{\\rm model}$ (Al 反サイト, $x-0.5$)')
ax.plot(x_grid, c_mace, '--', color='tab:blue', lw=2,
        label='$c_{\\rm vac}^{\\rm MLIP}$ (MACE 最安定)')
ax.plot(x_grid, c_total_1473, '-.', color='tab:orange', lw=2.5,
        label='$c_{\\rm total}^{\\rm hybrid}$ (1473 K, Boltzmann Va+Al$_{\\rm Ni}$)')
ax.plot(x_grid, c_total_1273, ':', color='tab:purple', lw=2.5,
        label='$c_{\\rm total}^{\\rm hybrid}$ (1273 K)')
ax.plot(x_grid, c_hyb_1473, '-.', color='tab:orange', lw=1.2, alpha=0.6,
        label='$c_{\\rm vac}^{\\rm hybrid}$ (1473 K, 空孔分)')
ax.scatter(td_al.x_Al_at / 100.0, td_al.c_vac_exp, color='tab:red', s=80, zorder=5,
           label='$c_{\\rm vac}^{\\rm exp}$ (T&D 密度, Table 2)', edgecolors='k', linewidths=0.5)

ax.axhline(0.0, color='gray', lw=1.0, ls='--')
ax.axvline(0.5, color='gray', lw=1.0, ls=':')
ax.set_xlabel('$x_{\\rm Al}$', fontsize=18)
ax.set_ylabel('Ni 副格子欠陥占有率（全サイト基準）', fontsize=16)
ax.set_title('B2-NiAl Al 過剰側：空孔・反サイト・Boltzmann 混合', fontsize=18)
ax.set_xlim(0.45, 0.60)
ax.set_ylim(-0.03, 0.22)
ax.legend(fontsize=9, loc='upper left')

# annotation: all models converge to perfect B2 at x=0.5
ax.text(0.51, 0.17,
        '化学量論組成 $x_{\\rm Al}=0.50$ では、\n'
        '完全 B2 が唯一の状態。\n'
        'Ni 空孔モデルと Al 反サイトモデルは\n'
        'ともに $c=0$ から分岐する。\n'
        '有限温度では 2 モデルが Boltzmann 混合し、\n'
        '$c_{\\rm total}^{\\rm hybrid}$ は 2 つの\n'
        '0 K 極限の間に位置する。',
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

out = os.path.join(FIG, 'fig_b2_vacancy_concentration.png')
plt.savefig(out, dpi=150)
plt.close()
print('Wrote', out)
