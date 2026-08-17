#!/usr/bin/env python3
"""Generate the full Ni-Al formation-energy diagram with a correct convex hull.

Includes MACE-relaxed pure elements, B2 defects, fcc-SQS, and intermetallic
compounds (L1_2-Ni_3Al, Ni_5Al_3, Ni_2Al_3, NiAl_3).  The convex hull is
computed from MACE formation energies on a per-occupied-atom basis, consistent
with the B2 off-stoichiometry analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..'))
AN_B2 = os.path.join(ROOT, 'b2_offstoich', 'analysis')
AN_NI = os.path.join(ROOT, 'niall_ext', 'analysis')

plt.rcParams.update({'font.size': 20, 'axes.grid': True, 'grid.alpha': 0.3,
                     'font.family': ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif'],
                     'axes.unicode_minus': False})

# --- pure-element and compound references (MACE-relaxed) ----------------------
mace_ref = pd.read_csv(os.path.join(AN_B2, 'mace_mp_ref_results.csv'))
mace_ref['x_Al'] = mace_ref.n_Al / mace_ref.n_atoms
compounds = mace_ref[mace_ref.label.isin(['L12_Ni3Al', 'Ni5Al3', 'Ni2Al3', 'NiAl3', 'B2_NiAl'])]

# --- fcc-SQS ------------------------------------------------------------------
sqs = pd.read_csv(os.path.join(AN_NI, 'niall_fcc_sqs.csv'))
mu_Ni = mace_ref[mace_ref.label == 'Ni'].energy_per_atom_eV.values[0]
mu_Al = mace_ref[mace_ref.label == 'Al'].energy_per_atom_eV.values[0]
sqs['Ef'] = (sqs.energy_eV / sqs.n_atoms) - ((1 - sqs.x_Al) * mu_Ni + sqs.x_Al * mu_Al)
sqs_plot = (sqs.groupby('x_Al')
             .agg(Ef=('Ef', 'mean'), Efstd=('Ef', 'std'))
             .reset_index())

# --- B2 off-stoichiometry branches -------------------------------------------
b2_files = [
    os.path.join(AN_B2, 'b2_offstoich_volumes.csv'),
    os.path.join(AN_B2, 'b2_offstoich_volumes_extra_vac.csv'),
    os.path.join(AN_B2, 'b2_offstoich_volumes_wide_vac.csv'),
    os.path.join(AN_B2, 'b2_offstoich_volumes_antisite_extra.csv'),
    os.path.join(AN_B2, 'b2_offstoich_volumes_50competition.csv'),
    os.path.join(AN_B2, 'b2_offstoich_volumes_antisite_alrich_dense.csv'),
]
b2 = pd.concat([pd.read_csv(f) for f in b2_files if os.path.exists(f)], ignore_index=True)


def agg_branch(df):
    return (df[df.branch != 'perfect'].groupby(['x_Al_target', 'branch'])
            .agg(x_Al=('x_Al', 'mean'), Ef=('E_form_eV_atom', 'mean'),
                 Efstd=('E_form_eV_atom', 'std')).reset_index())


b2_plot = agg_branch(b2)
perfect = b2[b2.branch == 'perfect'].iloc[0]

# --- convex hull (MACE points) -----------------------------------------------
points = []
# pure elements
points.append((0.0, 0.0))
points.append((1.0, 0.0))
# fcc-SQS
for _, r in sqs.iterrows():
    points.append((r.x_Al, r.Ef))
# B2 defect branches
for _, r in b2.iterrows():
    if r.branch == 'perfect':
        continue
    points.append((r.x_Al, r.E_form_eV_atom))
# compounds
for _, r in compounds.iterrows():
    points.append((r.x_Al, r.formation_energy_per_atom_eV))

pts = np.array(points)

# lower hull extraction
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

# --- plot --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 9))
ax.plot(pts[chain, 0], pts[chain, 1], 'k--', lw=2.5,
        label='凸包（0 K 安定性、L1$_2$/Ni$_5$Al$_3$/Ni$_2$Al$_3$/NiAl$_3$ 含む）')

# intermetallic compounds
comp_label = {'L12_Ni3Al': r"L1$_2$-Ni$_3$Al", 'Ni5Al3': r"Ni$_5$Al$_3$",
              'Ni2Al3': r"Ni$_2$Al$_3$", 'NiAl3': r"NiAl$_3$", 'B2_NiAl': r"B2-NiAl"}
for _, r in compounds.iterrows():
    ax.plot([r.x_Al], [r.formation_energy_per_atom_eV], 'D', ms=11,
            color='tab:green', zorder=6)
    ax.annotate(comp_label.get(r.label, r.label),
                xy=(r.x_Al, r.formation_energy_per_atom_eV),
                xytext=(0, 12), textcoords='offset points', fontsize=14,
                ha='center', zorder=7)

# fcc-SQS
ax.errorbar(sqs_plot.x_Al, sqs_plot.Ef, yerr=sqs_plot.Efstd, fmt='s-', ms=8,
            capsize=4, color='tab:purple', label='fcc-SQS Ni(Al) 固溶体', zorder=3)

# B2 branches
for br, g in b2_plot.groupby('branch'):
    g = g.sort_values('x_Al')
    col = 'tab:blue' if br == 'antisite' else 'tab:red'
    lab = 'B2 反サイト（antisite）配置' if br == 'antisite' else 'B2 空孔（vacancy）配置'
    ax.errorbar(g.x_Al, g.Ef, yerr=g.Efstd, fmt='o-', ms=9, capsize=4,
                color=col, label=lab, zorder=4)

# perfect B2 and pure elements
ax.plot([perfect.x_Al], [perfect.E_form_eV_atom], 's', ms=15, color='tab:orange',
        label='完全 B2-NiAl', zorder=5)
ax.plot([0.0, 1.0], [0.0, 0.0], 'ko', ms=10, label='純 fcc-Ni / 純 fcc-Al', zorder=5)

ax.set_xlabel(r"Al 原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"形成エネルギー $E_f$ (eV/atom)")
ax.set_title(r"Ni-Al 系：形成エネルギー図（中間化合物を含む正しい凸包）")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.75, 0.05)
ax.legend(fontsize=13, loc='lower right')
plt.tight_layout()
out = os.path.join(BASE, 'fig_energy_diagram_nial.png')
plt.savefig(out, dpi=150)
plt.close()
print('wrote', out)
