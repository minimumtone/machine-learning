#!/usr/bin/env python3
"""Overlay MACE lattice constant a(x) on Taylor & Doyle (1972) primary data.

Taylor & Doyle, J. Appl. Cryst. 5 (1972) 201, report linear lattice-parameter
fits for the beta-NiAl phase field; the NASA review TM-105598 (Noebe et al.)
explicitly gives the two regression ranges (Noebe et al., NASA Lewis, 1992):

  * Ni-rich side (50 <= at.% Ni <= 66)
      a [Å] = 2.8870 + (2.8618-2.8870)/(0.66-0.50) * (x_Ni - 0.50)
      with 2.00 atoms per unit cell.

  * Al-rich side (45 <= at.% Ni <= 50)
      a [Å] = 2.8870 + (2.8652-2.8870)/(0.45-0.50) * (x_Ni - 0.50)
      with the number of atoms per unit cell falling from 2.00 to 1.817
      at x_Al = 0.55 (45 at.% Ni).

The Al-rich fit is physically reliable only between x_Al = 0.50 and 0.55;
continuing it to 0.66 is an extrapolation.  The atom-per-cell count follows
the structural-vacancy model, n_cell = 1/x_Al, on the Al-rich side.

Here x_Al = 1 - x_Ni.  We compare the MACE lower-Helmholtz branch to these
reconstructed experimental lines and report slopes da/dx and dV/dx on the
Ni-rich and Al-rich sides of stoichiometry.
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

# --- Taylor & Doyle linear reconstruction ------------------------------------
def a_taylor_doyle(x_Al):
    """Taylor & Doyle (1972) primary linear a(x) reconstruction.

    Endpoint conventions:
      * 50 <= at.% Ni <= 66 : a = 2.8870 -> 2.8618 (Ni-rich, x_Al 0.50 -> 0.34)
      * 45 <= at.% Ni <= 50 : a = 2.8652 -> 2.8870 (Al-rich, x_Al 0.55 -> 0.50)

    The Al-rich endpoint 2.8652 Å corresponds to 45 at.% Ni (x_Al = 0.55),
    which is the Al-rich B2 single-phase limit seen by density measurements.
    Using the 34 at.% Ni (x_Al = 0.66) endpoint that appears in some older
    reproductions makes the Al-rich slope far too shallow.
    """
    x_Ni = 1.0 - x_Al
    if x_Al <= 0.5:
        a = 2.8870 + (2.8618 - 2.8870) / (0.66 - 0.50) * (x_Ni - 0.50)
        n_cell = 2.00
    else:
        a = 2.8870 + (2.8652 - 2.8870) / (0.45 - 0.50) * (x_Ni - 0.50)
        # structural-vacancy model: one atom per x_Al * a^3 unit cell
        # is equivalent to 1/x_Al atoms per B2 formula cell on the Al-rich side.
        n_cell = 1.0 / x_Al
    return a, n_cell

# --- MACE stable branch ------------------------------------------------------
mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
mask_b2 = (mix.x_Al >= 0.42) & (mix.x_Al <= 0.66)
mace = mix[mask_b2].copy().sort_values('x_Al')

# --- write reconstructed Taylor & Doyle a(x) table ---------------------------
points = [0.34, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.66]
td_rows = []
for x in points:
    a, n_cell = a_taylor_doyle(x)
    td_rows.append({'x_Al': x, 'a_A': a, 'n_atoms_per_cell': n_cell,
                    'V_per_atom_A3': a**3 / n_cell})
td = pd.DataFrame(td_rows)
td.to_csv(os.path.join(AN, 'taylor_doyle_a_reconstructed.csv'), index=False)
print(td.to_string(index=False))

# --- slope comparison --------------------------------------------------------
def slope(df, col, x0, x1):
    sub = df[(df.x_Al >= x0) & (df.x_Al <= x1)]
    if len(sub) < 2:
        return np.nan
    p = np.polyfit(sub.x_Al.values, sub[col].values, 1)
    return float(p[0])

# Slope comparisons use the same composition intervals for MACE and Taylor & Doyle.
# For Ni-rich we use 0.42-0.50; for Al-rich we show both 0.50-0.60 (closer to
# the reliable B2 single-phase data) and 0.50-0.66 (the full T&D reported range).
slope_mace_ni = slope(mace, 'a_mix', 0.42, 0.50)
slope_mace_al_60 = slope(mace, 'a_mix', 0.50, 0.60)
slope_mace_al_66 = slope(mace, 'a_mix', 0.50, 0.66)
slope_mace_v_ni = slope(mace, 'V_mix', 0.42, 0.50)
slope_mace_v_al_60 = slope(mace, 'V_mix', 0.50, 0.60)
slope_mace_v_al_66 = slope(mace, 'V_mix', 0.50, 0.66)

V42 = a_taylor_doyle(0.42)[0]**3 / a_taylor_doyle(0.42)[1]
V50 = a_taylor_doyle(0.50)[0]**3 / a_taylor_doyle(0.50)[1]
V60 = a_taylor_doyle(0.60)[0]**3 / a_taylor_doyle(0.60)[1]
V66 = a_taylor_doyle(0.66)[0]**3 / a_taylor_doyle(0.66)[1]

def slope_td_a(x0, x1):
    return (a_taylor_doyle(x1)[0] - a_taylor_doyle(x0)[0]) / (x1 - x0)
def slope_td_v(x0, x1):
    return (a_taylor_doyle(x1)[0]**3 / a_taylor_doyle(x1)[1]
            - a_taylor_doyle(x0)[0]**3 / a_taylor_doyle(x0)[1]) / (x1 - x0)

slope_td_ni = slope_td_a(0.42, 0.50)
slope_td_al_60 = slope_td_a(0.50, 0.60)
slope_td_al_66 = slope_td_a(0.50, 0.66)
slope_td_v_ni = slope_td_v(0.42, 0.50)
slope_td_v_al_60 = slope_td_v(0.50, 0.60)
slope_td_v_al_66 = slope_td_v(0.50, 0.66)

slope_table = pd.DataFrame([
    {'side': 'Ni-rich (0.42-0.50)', 'quantity': 'da/dx', 'MACE': slope_mace_ni, 'Taylor_Doyle': slope_td_ni, 'unit': 'Å / x_Al', 'MACE/TD_ratio': slope_mace_ni/slope_td_ni},
    {'side': 'Al-rich (0.50-0.60)', 'quantity': 'da/dx', 'MACE': slope_mace_al_60, 'Taylor_Doyle': slope_td_al_60, 'unit': 'Å / x_Al', 'MACE/TD_ratio': slope_mace_al_60/slope_td_al_60 if abs(slope_td_al_60)>1e-12 else np.nan},
    {'side': 'Al-rich (0.50-0.66)', 'quantity': 'da/dx', 'MACE': slope_mace_al_66, 'Taylor_Doyle': slope_td_al_66, 'unit': 'Å / x_Al', 'MACE/TD_ratio': slope_mace_al_66/slope_td_al_66 if abs(slope_td_al_66)>1e-12 else np.nan},
    {'side': 'Ni-rich (0.42-0.50)', 'quantity': 'dV/dx', 'MACE': slope_mace_v_ni, 'Taylor_Doyle': slope_td_v_ni, 'unit': 'Å3/atom / x_Al', 'MACE/TD_ratio': slope_mace_v_ni/slope_td_v_ni},
    {'side': 'Al-rich (0.50-0.60)', 'quantity': 'dV/dx', 'MACE': slope_mace_v_al_60, 'Taylor_Doyle': slope_td_v_al_60, 'unit': 'Å3/atom / x_Al', 'MACE/TD_ratio': slope_mace_v_al_60/slope_td_v_al_60 if abs(slope_td_v_al_60)>1e-12 else np.nan},
    {'side': 'Al-rich (0.50-0.66)', 'quantity': 'dV/dx', 'MACE': slope_mace_v_al_66, 'Taylor_Doyle': slope_td_v_al_66, 'unit': 'Å3/atom / x_Al', 'MACE/TD_ratio': slope_mace_v_al_66/slope_td_v_al_66 if abs(slope_td_v_al_66)>1e-12 else np.nan},
])
slope_table.to_csv(os.path.join(AN, 'taylor_doyle_mace_slopes.csv'), index=False)
print(slope_table.to_string(index=False))

# --- point-by-point comparison at experimental x values -----------------------
comp = []
for _, r in td.iterrows():
    if r.x_Al < 0.42 - 1e-9 or r.x_Al > 0.66 + 1e-9:
        continue
    if mace.x_Al.min() - 1e-9 <= r.x_Al <= mace.x_Al.max() + 1e-9:
        a_mace = float(np.interp(r.x_Al, mace.x_Al.values, mace.a_mix.values))
        comp.append({
            'x_Al': r.x_Al,
            'a_Taylor_Doyle_A': round(r.a_A, 4),
            'a_MACE_A': round(a_mace, 4),
            'delta_A': round(a_mace - r.a_A, 4),
        })
pd.DataFrame(comp).to_csv(os.path.join(AN, 'a_comparison_taylor_doyle_mace.csv'), index=False)

# --- plot --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(mace.x_Al, mace.a_mix, 'o-', color='tab:blue', ms=7,
        label='MACE 安定枝 ($G$ per atom)')

x_fine = np.linspace(0.34, 0.66, 200)
a_fine = [a_taylor_doyle(xx)[0] for xx in x_fine]
ax.plot(x_fine, a_fine, '--', color='tab:orange', lw=2,
        label='Taylor & Doyle (1972) 一次再構成')

ax.axvline(0.5, color='gray', ls=':', lw=1)
ax.set_xlabel(r"Al 原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"格子定数 $a$ (Å)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 格子定数：MACE vs Taylor & Doyle")
ax.legend(fontsize=12)
ax.set_xlim(0.32, 0.68)
ax.set_ylim(2.82, 2.92)
plt.tight_layout()
out = os.path.join(FIG, 'fig_b2_a_taylor_doyle_overlay.png')
plt.savefig(out, dpi=150)
plt.close()
print('Wrote', out)
