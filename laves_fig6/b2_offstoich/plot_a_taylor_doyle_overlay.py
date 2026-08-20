#!/usr/bin/env python3
"""Overlay MACE lattice constant a(x) on Taylor & Doyle (1972) Table 2 raw data.

Taylor & Doyle (J. Appl. Cryst. 5 (1972) 201) measured both X-ray lattice
parameters and Archimedes densities for beta-NiAl phase alloys.  Table 2 gives
the directly measured a, density rho and derived number of atoms per unit cell
n.  From these we obtain the experimental average atomic volume

    Vbar = a**3 / n   (A^3 / atom)

and the structural vacancy concentration on the Al-rich side,

    c_vac = 1 - n / 2.

The script compares the MACE lower-Helmholtz branch with the raw Taylor & Doyle
points and reports local slopes da/dx and dVbar/dx on the Ni-rich and Al-rich
sides of stoichiometry.
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
td = td.sort_values('x_Al_at')

# --- MACE stable branch --------------------------------------------------------
mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
mask_b2 = (mix.x_Al >= 0.35) & (mix.x_Al <= 0.66)
mace = mix[mask_b2].copy().sort_values('x_Al')

# --- slope comparison (use the raw T&D points and the MACE branch) --------------
def slope(df, col, x0, x1):
    sub = df[(df.x_Al >= x0) & (df.x_Al <= x1)]
    if len(sub) < 2:
        return np.nan
    p = np.polyfit(sub.x_Al.values, sub[col].values, 1)
    return float(p[0])

def slope_td(df, col, x0, x1):
    sub = df[(df.x_Al_at >= x0) & (df.x_Al_at <= x1) & df[col].notna()]
    if len(sub) < 2:
        return np.nan
    p = np.polyfit(sub.x_Al_at.values / 100.0, sub[col].values, 1)
    return float(p[0])

# Robust intervals: Al-rich x_Al = 0.50-0.55 (50-55 at.% Al), x_Ni = 45-50 at.% Ni
#                   Ni-rich x_Al = 0.45-0.50 (45-50 at.% Al), x_Ni = 50-55 at.% Ni
slope_mace_ni = slope(mace, 'a_mix', 0.45, 0.50)
slope_mace_al = slope(mace, 'a_mix', 0.50, 0.55)
slope_mace_v_ni = slope(mace, 'V_mix', 0.45, 0.50)
slope_mace_v_al = slope(mace, 'V_mix', 0.50, 0.55)

slope_td_ni = slope_td(td, 'a_A', 45.0, 50.0)
slope_td_al = slope_td(td, 'a_A', 50.0, 55.0)
slope_td_v_ni = slope_td(td, 'Vbar_A3_per_atom', 45.0, 50.0)
slope_td_v_al = slope_td(td, 'Vbar_A3_per_atom', 50.0, 55.0)

slope_table = pd.DataFrame([
    {'side': 'Ni-rich (0.45-0.50)', 'quantity': 'da/dx', 'MACE': slope_mace_ni,
     'Taylor_Doyle': slope_td_ni, 'unit': 'A / x_Al',
     'MACE/TD_ratio': slope_mace_ni / slope_td_ni if abs(slope_td_ni) > 1e-12 else np.nan},
    {'side': 'Al-rich (0.50-0.55)', 'quantity': 'da/dx', 'MACE': slope_mace_al,
     'Taylor_Doyle': slope_td_al, 'unit': 'A / x_Al',
     'MACE/TD_ratio': slope_mace_al / slope_td_al if abs(slope_td_al) > 1e-12 else np.nan},
    {'side': 'Ni-rich (0.45-0.50)', 'quantity': 'dVbar/dx', 'MACE': slope_mace_v_ni,
     'Taylor_Doyle': slope_td_v_ni, 'unit': 'A3/atom / x_Al',
     'MACE/TD_ratio': slope_mace_v_ni / slope_td_v_ni if abs(slope_td_v_ni) > 1e-12 else np.nan},
    {'side': 'Al-rich (0.50-0.55)', 'quantity': 'dVbar/dx', 'MACE': slope_mace_v_al,
     'Taylor_Doyle': slope_td_v_al, 'unit': 'A3/atom / x_Al',
     'MACE/TD_ratio': slope_mace_v_al / slope_td_v_al if abs(slope_td_v_al) > 1e-12 else np.nan},
])
slope_table.to_csv(os.path.join(AN, 'taylor_doyle_mace_slopes.csv'), index=False)
print(slope_table.to_string(index=False))

# --- point-by-point comparison at experimental x values ------------------------
comp = []
for _, r in td.iterrows():
    x = r.x_Al_at / 100.0
    if mace.x_Al.min() - 1e-9 <= x <= mace.x_Al.max() + 1e-9 and not np.isnan(r.a_A):
        a_mace = float(np.interp(x, mace.x_Al.values, mace.a_mix.values))
        v_mace = float(np.interp(x, mace.x_Al.values, mace.V_mix.values))
        comp.append({
            'x_Al': round(x, 4),
            'a_Taylor_Doyle_A': round(r.a_A, 4),
            'a_MACE_A': round(a_mace, 4),
            'delta_A': round(a_mace - r.a_A, 4),
            'Vbar_Taylor_Doyle_A3': round(r.Vbar_A3_per_atom, 3) if not np.isnan(r.Vbar_A3_per_atom) else None,
            'Vbar_MACE_A3': round(v_mace, 3),
        })
pd.DataFrame(comp).to_csv(os.path.join(AN, 'a_comparison_taylor_doyle_mace.csv'), index=False)
print(pd.DataFrame(comp).to_string(index=False))

# --- plot ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(mace.x_Al, mace.a_mix, '-', color='tab:blue', lw=2,
        label='MACE 安定モデル ($G$ per atom)')

al = td[td.x_Al_at > 50.0]
ni = td[td.x_Al_at <= 50.0]
ax.scatter(al.x_Al_at / 100.0, al.a_A, color='tab:orange', s=80, zorder=5,
           label='Taylor & Doyle (1972) Al-rich', edgecolors='k', linewidths=0.5)
ax.scatter(ni.x_Al_at / 100.0, ni.a_A, color='tab:green', s=80, zorder=5,
           label='Taylor & Doyle (1972) Ni-rich', edgecolors='k', linewidths=0.5, marker='s')

# linear fits to raw points
al_fit = al[al.a_A.notna()].sort_values('x_Al_at')
ni_fit = ni[(ni.a_A.notna()) & (ni.x_Al_at <= 56.0)].sort_values('x_Al_at')
if len(al_fit) >= 2:
    p = np.polyfit(al_fit.x_Al_at.values / 100.0, al_fit.a_A.values, 1)
    x_fit = np.linspace(0.50, 0.55, 100)
    ax.plot(x_fit, np.polyval(p, x_fit), '--', color='tab:orange', lw=1.5, alpha=0.7,
            label=f'T&D Al-rich fit: da/dx = {p[0]:.3f} A/x_Al')
if len(ni_fit) >= 2:
    p = np.polyfit(ni_fit.x_Al_at.values / 100.0, ni_fit.a_A.values, 1)
    x_fit = np.linspace(0.43, 0.50, 100)
    ax.plot(x_fit, np.polyval(p, x_fit), '--', color='tab:green', lw=1.5, alpha=0.7,
            label=f'T&D Ni-rich fit: da/dx = {p[0]:.3f} A/x_Al')

ax.axvline(0.5, color='gray', ls=':', lw=1)
ax.set_xlabel(r"$x_{\mathrm{Al}}$")
ax.set_ylabel(r"格子定数 $a$ (A)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 格子定数：MACE vs Taylor & Doyle (Table 2 raw)")
ax.legend(fontsize=11, loc='best')
ax.set_xlim(0.35, 0.58)
ax.set_ylim(2.84, 2.90)
plt.tight_layout()
out = os.path.join(FIG, 'fig_b2_a_taylor_doyle_overlay.png')
plt.savefig(out, dpi=150)
plt.close()
print('Wrote', out)
