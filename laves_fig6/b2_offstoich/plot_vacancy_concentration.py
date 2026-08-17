#!/usr/bin/env python3
"""Experimental proof of structural vacancies on B2-NiAl from a(x) and Vbar(x).

Taylor & Doyle (1972) give the lattice constant a(x) of the beta-NiAl phase
field.  Yamanouchi & Miura / Ellner give the atomic volume Vbar(x) (digitised
from Fig. 6(a)).  Because these two measurements are independent, their
combination yields the structural vacancy concentration directly:

    c_vac(x) = 1 - a(x)^3 / (2 * Vbar(x))

This is compared to (i) the single-vacancy model for Ni vacancies on the Ni
sublattice,

    c_vac_model(x) = (2 - 1/x) / 2 = 1 - 1/(2x)

and (ii) the value obtained from MACE-MP-0 relaxed a(x) and Vbar(x).

The antisite-only model would predict c_vac = 0 and Vbar = a^3/2 everywhere,
which is clearly contradicted by the experimental data for x_Al > 0.50.
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


def a_taylor_doyle(x_Al):
    """Taylor & Doyle (1972) primary linear a(x) reconstruction."""
    x_Ni = 1.0 - x_Al
    if x_Al <= 0.5:
        a = 2.8870 + (2.8618 - 2.8870) / (0.66 - 0.50) * (x_Ni - 0.50)
    else:
        a = 2.8870 + (2.8652 - 2.8870) / (0.34 - 0.50) * (x_Ni - 0.50)
    return a


# --- experimental Vbar from digitised Yamanouchi Fig. 6(a) -------------------
exp_b2 = pd.read_csv(os.path.join(AN, 'fig6a_digitized_circles.csv'))
exp_b2 = exp_b2[(exp_b2.region == 'B2') & (exp_b2.x_Al >= 0.45) & (exp_b2.x_Al <= 0.62)].copy()
# smooth by bin-averaging; window = 0.02
bin_width = 0.02
bins = np.arange(0.44, 0.62 + bin_width, bin_width)
exp_b2['bin'] = pd.cut(exp_b2.x_Al, bins)
vbar_binned = exp_b2.groupby('bin', observed=False).agg({'x_Al': 'mean', 'V_bar_A3': 'mean'}).dropna()
vbar_binned = vbar_binned.reset_index(drop=True)

x_grid = np.linspace(0.45, 0.60, 200)
Vbar_exp = np.interp(x_grid, vbar_binned.x_Al, vbar_binned.V_bar_A3)

# --- model structural vacancy concentration ----------------------------------
def c_vac_model(x):
    return np.clip((2.0 - 1.0 / np.asarray(x, dtype=float)) / 2.0, 0.0, 1.0)

c_model = c_vac_model(x_grid)

# --- experimental c_vac from independent a(x) and Vbar -----------------------
a_exp_grid = np.array([a_taylor_doyle(x) for x in x_grid])
c_exp = 1.0 - a_exp_grid**3 / (2.0 * Vbar_exp)
c_exp = np.clip(c_exp, 0.0, 1.0)

# --- MACE a(x) and Vbar ------------------------------------------------------
mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
mask = (mix.x_Al >= 0.45) & (mix.x_Al <= 0.66)
mace = mix[mask].sort_values('x_Al')
a_mace = np.interp(x_grid, mace.x_Al.values, mace.a_mix.values)
V_mace = np.interp(x_grid, mace.x_Al.values, mace.V_mix.values)
c_mace = 1.0 - a_mace**3 / (2.0 * V_mace)
c_mace = np.clip(c_mace, 0.0, 1.0)

# antisite-only model: c=0

# --- output table ------------------------------------------------------------
table_x = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
table = []
for x in table_x:
    a = a_taylor_doyle(x)
    V = float(np.interp(x, vbar_binned.x_Al, vbar_binned.V_bar_A3))
    table.append({
        'x_Al': x,
        'a_TD_A': round(a, 4),
        'Vbar_exp_A3': round(V, 3),
        'c_vac_exp': round(float(1.0 - a**3 / (2.0 * V)), 3),
        'c_vac_model': round(float(c_vac_model(x)), 3),
        'c_vac_MACE': round(float(np.interp(x, x_grid, c_mace)), 3),
    })
table = pd.DataFrame(table)
table.to_csv(os.path.join(AN, 'vacancy_concentration_exp_vs_mace.csv'), index=False)
print('--- c_vac (exp / model / MACE) ---')
print(table.to_string(index=False))

# --- plot --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(x_grid, c_model, 'k-', lw=2.5, label='$c_{\\rm vac}^{\\rm model}$ (Ni 空孔, $=(2-1/x)/2$)')
ax.plot(x_grid, c_exp, 'o-', color='tab:orange', ms=4, markevery=12, lw=2,
        label='$c_{\\rm vac}^{\\rm exp}$ (Taylor & Doyle $a(x)$ + Ellner/Yamanouchi $\\bar V(x)$)')
if len(mace) > 0:
    ax.plot(x_grid, c_mace, 's--', color='tab:blue', ms=4, markevery=12, lw=2,
            label='$c_{\\rm vac}^{\\rm MACE}$ (MACE $a(x)$ + $\\bar V(x)$)')

# mark antisite-only prediction
ax.axhline(0.0, color='tab:gray', lw=1.0, ls='--', label='反サイトのみモデル ($c_{\\rm vac}=0$)')

# mark key compositions
ax.axvline(0.5, color='tab:green', lw=1.0, ls=':')
ax.axvline(2.0/3.0, color='tab:green', lw=1.0, ls=':')
ax.text(0.505, 0.28, 'B2 化学量論', fontsize=11, color='tab:green')
ax.text(0.62, 0.28, 'NiAl$_{2}$-limit ($x=2/3$)', fontsize=11, color='tab:green')

ax.set_xlabel('$x_{\\rm Al}$', fontsize=18)
ax.set_ylabel('構成空孔分率 $c_{\\rm vac}$', fontsize=18)
ax.set_title('B2-NiAl 構成空孔濃度：独立実験（$a$ + $\\bar V$）の直接証明', fontsize=18)
ax.set_xlim(0.45, 0.61)
ax.set_ylim(-0.03, 0.22)
ax.legend(fontsize=13, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_b2_vacancy_concentration.png'), dpi=150)
plt.close()
print('Wrote', os.path.join(FIG, 'fig_b2_vacancy_concentration.png'))
