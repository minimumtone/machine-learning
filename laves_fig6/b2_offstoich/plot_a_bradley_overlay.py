#!/usr/bin/env python3
"""Overlay MACE lattice constant a(x) on Bradley & Taylor experimental data.

Bradley & Taylor data points are read approximately from the reproduction in
Jiang & Chen, Acta Mater. 53 (2005) Fig. 4(a).  x is converted from Ni fraction
to Al fraction (x_Al = 1 - x_Ni).
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, 'analysis')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

# MACE stable-branch data: use the Boltzmann mix (per-atom free energy) where
# it has two branches, otherwise fallback to the lower-of-two 0 K branch.
mix = pd.read_csv(os.path.join(AN, 'b2_offstoich_boltzmann_mix.csv'))
# restrict to physically meaningful B2 range (drop the oscillating Al-rich tail)
mace = mix[(mix.x_Al >= 0.42) & (mix.x_Al <= 0.66)].copy()
mace['source'] = 'MACE stable branch'

# Bradley & Taylor data: x_Ni -> x_Al = 1 - x_Ni
bradley = pd.DataFrame({
    'x_Ni': [0.40, 0.45, 0.50, 0.52, 0.55, 0.60],
    'a_A':  [2.87, 2.87, 2.88, 2.86, 2.86, 2.83],
})
# the x_Ni=0.60 (x_Al=0.40) point sits at the Al-rich edge of the B2 field and
# may include incipient Ni2Al3 in the older measurements; flag it.
bradley['x_Al'] = 1.0 - bradley['x_Ni']
bradley['note'] = ['edge' if x < 0.45 else 'B2' for x in bradley['x_Al']]

# write experimental CSV
bradley[['x_Al','a_A','note']].to_csv(os.path.join(AN,'bradley_taylor_a_exp.csv'), index=False)

# write comparison table: MACE a nearest to experimental x
mace_interp = mace.set_index('x_Al').a_mix
rows = []
for _, r in bradley.iterrows():
    if abs(r.x_Al - 0.5) <= 0.11:  # within plotted range
        a_mace = float(np.interp(r.x_Al, mace.x_Al.values, mace.a_mix.values))
        rows.append({
            'x_Al': r.x_Al,
            'a_Bradley_Taylor_A': r.a_A,
            'a_MACE_A': round(a_mace, 4),
            'delta_A': round(a_mace - r.a_A, 4),
            'note': r.note,
        })
pd.DataFrame(rows).to_csv(os.path.join(AN, 'a_comparison_bradley_mace.csv'), index=False)

# plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(mace.x_Al, mace.a_mix, 'o-', color='tab:blue', ms=7,
        label=r'MACE 安定枝 ($\Omega$ per atom)')
mask = bradley.note == 'B2'
ax.errorbar(bradley[mask].x_Al, bradley[mask].a_A, yerr=0.01, fmt='^', ms=10,
            color='tab:orange', capsize=4, zorder=5, label='Bradley & Taylor (1937) B2 領域')
if (~mask).any():
    ax.errorbar(bradley[~mask].x_Al, bradley[~mask].a_A, yerr=0.01, fmt='v', ms=10,
                mfc='none', mec='tab:orange', mew=2, capsize=4, zorder=5,
                label='Bradley & Taylor (edge / two-phase)')

ax.axvline(0.5, color='gray', ls=':', lw=1)
ax.set_xlabel(r"Al 原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"格子定数 $a$ (\AA)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 格子定数：MACE vs Bradley \& Taylor")
ax.legend(fontsize=12)
ax.set_xlim(0.37, 0.63)
ax.set_ylim(2.80, 2.95)
plt.tight_layout()
out = os.path.join(FIG, 'fig_b2_a_bradley_overlay.png')
plt.savefig(out, dpi=150)
plt.close()
print('Wrote', out)
print(pd.DataFrame(rows).to_string(index=False))
