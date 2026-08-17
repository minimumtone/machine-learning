#!/usr/bin/env python3
"""Analyze Al-rich dense antisite data for two-phase separation tendency."""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..'))
AN = os.path.join(BASE, 'analysis')

mu = pd.read_csv(os.path.join(AN, 'mace_mp_ref_results.csv'))
mu_Ni = float(mu[mu.label == 'Ni'].energy_per_atom_eV.values[0])
mu_Al = float(mu[mu.label == 'Al'].energy_per_atom_eV.values[0])

def ef(row):
    return row.energy_eV / row.n_atoms - ((1 - row.x_Al) * mu_Ni + row.x_Al * mu_Al)

b2_files = glob.glob(os.path.join(AN, 'b2_offstoich_volumes*.csv'))
b2 = pd.concat([pd.read_csv(f) for f in b2_files if os.path.exists(f)], ignore_index=True)
b2['Ef'] = b2.apply(ef, axis=1)
b2c = b2[b2.converged].copy()

sqs = pd.read_csv(os.path.join(ROOT, 'niall_ext', 'analysis', 'niall_fcc_sqs.csv'))
sqs['Ef'] = sqs.energy_eV / sqs.n_atoms - ((1 - sqs.x_Al) * mu_Ni + sqs.x_Al * mu_Al)

comp = mu[mu.label.isin(['L12_Ni3Al', 'Ni5Al3', 'Ni2Al3', 'NiAl3', 'B2_NiAl'])].copy()
comp['x_Al'] = comp.n_Al / comp.n_atoms
comp['Ef'] = comp.formation_energy_per_atom_eV

points = [(0.0, 0.0), (1.0, 0.0)]
for _, r in b2c.iterrows():
    points.append((r.x_Al, r.Ef))
for _, r in sqs.iterrows():
    points.append((r.x_Al, r.Ef))
for _, r in comp.iterrows():
    points.append((r.x_Al, r.Ef))
pts = np.array(points)

hull = ConvexHull(pts)
edges = []
for i, j in hull.simplices:
    x1, y1 = pts[i]; x2, y2 = pts[j]
    if x1 > x2: i, j = j, i; x1, y1 = pts[i]; x2, y2 = pts[j]
    if abs(x2 - x1) < 1e-9: continue
    m = (y2 - y1) / (x2 - x1)
    ok = True
    for k in range(len(pts)):
        if k == i or k == j: continue
        xx, yy = pts[k]
        if min(x1, x2) - 1e-9 <= xx <= max(x1, x2) + 1e-9:
            if yy < m * (xx - x1) + y1 - 1e-7:
                ok = False; break
    if ok: edges.append((i, j))

adj = {}
for i, j in edges:
    adj.setdefault(i, set()).add(j)
    adj.setdefault(j, set()).add(i)
start = int(np.argmin(pts[:, 0]))
seen = set(); chain = []
def walk(i):
    seen.add(i); chain.append(i)
    nx = [j for j in adj.get(i, []) if j not in seen and pts[j, 0] >= pts[i, 0] - 1e-9]
    if nx: walk(min(nx, key=lambda k: pts[k, 0]))
walk(start)
chain = sorted(chain, key=lambda i: pts[i, 0])
xh, yh = pts[chain][:, 0], pts[chain][:, 1]

dense = b2[b2.branch == 'antisite'].copy()
dense = dense[(dense.x_Al_target >= 0.60) & (dense.x_Al_target <= 0.80)]
agg = (dense.groupby('x_Al_target')
       .agg(x_Al=('x_Al', 'mean'), Ef=('Ef', 'mean'), Efstd=('Ef', 'std'),
            V=('V_per_atom_A3', 'mean'), Vstd=('V_per_atom_A3', 'std'),
            conv=('converged', 'all'))
       .reset_index())
agg['Ef_hull'] = np.interp(agg.x_Al, xh, yh)
agg['delta_above_hull'] = agg.Ef - agg.Ef_hull
agg.to_csv(os.path.join(AN, 'b2_alrich_dense_phase_stability.csv'), index=False)

plt.rcParams.update({'font.size': 18, 'axes.grid': True, 'grid.alpha': 0.3,
                     'font.family': ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif']})
fig, ax = plt.subplots(figsize=(11, 7))
ax.plot(xh, yh, 'k--', lw=2, label='凸包（MACE, 0 K）')
ax.scatter(b2c.x_Al, b2c.Ef, c='gray', s=20, alpha=0.4, zorder=1)
ax.errorbar(agg.x_Al, agg.Ef, yerr=agg.Efstd, fmt='ro', ms=9,
            capsize=4, label='Al-rich 反サイト密サンプリング')
ax.set_xlabel(r'Al 原子分率 $x_{\rm Al}$')
ax.set_ylabel(r'形成エネルギー $E_f$ (eV/atom)')
ax.set_title('Al 過剰 B2 反サイト枝の凸包からの乖離')
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'figures', 'fig_b2_alrich_dense_hull.png'), dpi=150)
print(f"Wrote {os.path.join(AN, 'b2_alrich_dense_phase_stability.csv')}")
print(agg[['x_Al_target', 'x_Al', 'Ef', 'Ef_hull', 'delta_above_hull']].to_string(index=False))
