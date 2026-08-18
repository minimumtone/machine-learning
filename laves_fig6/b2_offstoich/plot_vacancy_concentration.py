#!/usr/bin/env python3
"""Experimental proof of structural vacancies on B2-NiAl from T&D density data.

Taylor & Doyle (1972, J. Appl. Cryst. 5, 201) measured both lattice parameters
and Archimedes densities for beta-NiAl alloys.  Table 2 gives the number of atoms
per unit cell n.  Because the conventional B2 cell contains two sites,

    c_vac = 1 - n / 2

is the vacancy fraction per lattice site on the Al-rich side.  This is an
independent experimental quantity: a(x) comes from X-ray diffraction, n comes from
density (mass/volume) and the same a.  The result is compared to the structural
vacancy model

    c_vac_model(x) = 1 - 1 / (2 x)   (x = x_Al)

and to MACE-MP-0 relaxed a(x) and Vbar(x).
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

# --- structural vacancy model --------------------------------------------------
def c_vac_model(x):
    return 1.0 - 1.0 / (2.0 * np.asarray(x, dtype=float))

# --- MACE stable branch --------------------------------------------------------
mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
mask = (mix.x_Al >= 0.45) & (mix.x_Al <= 0.66)
mace = mix[mask].sort_values('x_Al')

x_grid = np.linspace(0.45, 0.60, 200)
c_model = c_vac_model(x_grid)

a_mace = np.interp(x_grid, mace.x_Al.values, mace.a_mix.values)
V_mace = np.interp(x_grid, mace.x_Al.values, mace.V_mix.values)
N_mace = a_mace**3 / V_mace
c_mace = 1.0 - N_mace / 2.0
c_mace = np.clip(c_mace, 0.0, 1.0)

# --- output table --------------------------------------------------------------
table = []
for _, r in td_al.iterrows():
    x = r.x_Al_at / 100.0
    cmod = float(c_vac_model(x))
    # MACE at this x
    a_m = float(np.interp(x, mace.x_Al.values, mace.a_mix.values))
    V_m = float(np.interp(x, mace.x_Al.values, mace.V_mix.values))
    N_m = a_m**3 / V_m
    c_m = float(np.clip(1.0 - N_m / 2.0, 0.0, 1.0))
    table.append({
        'x_Al': round(x, 4),
        'a_TD_A': round(r.a_A, 4),
        'rho_gcm3': r.rho_gcm3,
        'n_atoms_per_cell': round(r.n_atoms_per_cell, 3),
        'c_vac_exp': round(r.c_vac_exp, 4),
        'c_vac_model': round(cmod, 4),
        'c_vac_MLIP': round(c_m, 4),
        'p_Al_antisite': round(max(0.0, x * r.n_atoms_per_cell - 1.0), 4),
    })
table = pd.DataFrame(table)
table.to_csv(os.path.join(AN, 'vacancy_concentration_exp_vs_mace.csv'), index=False)
print(table.to_string(index=False))

# --- plot ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(x_grid, c_model, 'k-', lw=2.5, label='$c_{\\rm vac}^{\\rm model}$ (Ni site vacancies, $1-1/(2x)$)')
ax.plot(x_grid, c_mace, '--', color='tab:blue', lw=2,
        label='$c_{\\rm vac}^{\\rm MLIP}$ (MACE selected branch)')
ax.scatter(td_al.x_Al_at / 100.0, td_al.c_vac_exp, color='tab:red', s=80, zorder=5,
           label='$c_{\\rm vac}^{\\rm exp}$ (T&D density, Table 2)', edgecolors='k', linewidths=0.5)

ax.axhline(0.0, color='gray', lw=1.0, ls='--')
ax.axvline(0.5, color='gray', lw=1.0, ls=':')
ax.set_xlabel('$x_{\\rm Al}$', fontsize=18)
ax.set_ylabel('構成空孔分率 $c_{\\rm vac}$', fontsize=18)
ax.set_title('B2-NiAl 構成空孔濃度：Taylor & Doyle 密度測定 vs MACE', fontsize=18)
ax.set_xlim(0.45, 0.60)
ax.set_ylim(-0.03, 0.22)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()
out = os.path.join(FIG, 'fig_b2_vacancy_concentration.png')
plt.savefig(out, dpi=150)
plt.close()
print('Wrote', out)
