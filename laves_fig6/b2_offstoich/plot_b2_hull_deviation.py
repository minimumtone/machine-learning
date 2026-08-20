#!/usr/bin/env python3
"""Plot B2 branch deviation from the 0 K and 1273 K reference hulls.

The first panel shows E_f and G(1273 K) relative to the 1273 K solid-phase hull
(Ni, Ni3Al, B2, Ni2Al3).  The second panel overlays the 0 K deviation gap and
-T*S_conf so that the finite-temperature crossing is visible.
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


def mark_perfect_b2(ax, x=0.5, y=0.0, label='完全B2'):
    """Composite red/blue/green marker showing vacancy and antisite branches
    coincide with the perfect B2 state at the stoichiometric composition."""
    ax.plot([x], [y], 's', ms=16, color='tab:green', zorder=8, label=label)
    ax.plot([x], [y], 'o', ms=12, mfc='none', mec='tab:red', mew=2.0,
            zorder=9, label='_nolegend_')
    ax.plot([x], [y], 'o', ms=7, mfc='tab:blue', mec='k', mew=0.5,
            zorder=10, label='_nolegend_')

ba = pd.read_csv(os.path.join(AN, 'b2_branch_finiteT_hull.csv'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = {'antisite': 'tab:blue', 'vacancy': 'tab:red', 'perfect': 'tab:orange'}

# panel 1: delta to 1273 K hull
for br, g in ba.groupby('branch'):
    if br == 'perfect':
        continue
    g = g.sort_values('x_Al')
    ax1.plot(g.x_Al, g['delta_Ef_1273K_eV'], 'o--', color=colors[br], ms=6, label=f'{br} $\\Delta E_f$')
    ax1.plot(g.x_Al, g['delta_G_1273K_1273K_eV'], 'o-', color=colors[br], ms=6, label=f'{br} $\\Delta G(1273)$')

ax1.axhline(0.0, color='black', lw=0.8)
ax1.axvline(0.5, color='tab:green', ls=':')
mark_perfect_b2(ax1)
ax1.set_xlabel('$x_{\\rm Al}$')
ax1.set_ylabel('凸包からの偏差 (eV / atom)')
ax1.set_title('B2 欠陥モデル vs 1273 K 凸包（Ni, Ni$_3$Al, B2, Ni$_2$Al$_3$）')
ax1.set_xlim(0.45, 0.67)
ax1.legend(fontsize=10)

# panel 2: 0 K gap and -TS for the vacancy branch
gv = ba[ba.branch == 'vacancy'].sort_values('x_Al')
ax2.plot(gv.x_Al, gv['delta_Ef_0K_all_eV'], 'o--', color='tab:red', label='空孔モデル $\\Delta E_f$ (0 K 全相凸包)')
ax2.plot(gv.x_Al, -gv['minus_T_S_1273K'], 's-', color='tab:green', label='-$T S_{\\rm conf}$ (1273 K)')
ax2.plot(gv.x_Al, gv['delta_G_1273K_1273K_eV'], 'o-', color='tab:blue', label='$\\Delta G = \\Delta E_f - T S_{\\rm conf}$')
ax2.axhline(0.0, color='black', lw=0.8)
ax2.axvline(0.5, color='tab:green', ls=':')
mark_perfect_b2(ax2)
ax2.set_xlabel('$x_{\\rm Al}$')
ax2.set_ylabel('eV / atom')
ax2.set_title('0 K 乖離と配置エントロピーの重ね合わせ（空孔モデル）')
ax2.set_xlim(0.48, 0.67)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_b2_hull_deviation.png'), dpi=150)
plt.close()
print('Wrote', os.path.join(FIG, 'fig_b2_hull_deviation.png'))
