#!/usr/bin/env python3
"""B2 branch convex-hull analysis with x_max extraction.

Includes Ni3Al4, L1_2, Ni5Al3, Ni2Al3, NiAl3, B2-NiAl on the MACE 0 K hull,
then reports how far each B2 branch point lies above that hull.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, 'analysis')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

# --- load references ---------------------------------------------------------
mace_ref = pd.read_csv(os.path.join(AN, 'mace_mp_ref_results.csv'))
mace_ref['x_Al'] = mace_ref.n_Al / mace_ref.n_atoms
mu_Ni = mace_ref[mace_ref.label == 'Ni'].energy_per_atom_eV.values[0]
mu_Al = mace_ref[mace_ref.label == 'Al'].energy_per_atom_eV.values[0]

compounds = mace_ref[mace_ref.label.isin(
    ['L12_Ni3Al', 'Ni3Al4', 'Ni5Al3', 'Ni2Al3', 'NiAl3', 'B2_NiAl'])]

# pure elements have E_f = 0 by construction for hull
pure = pd.DataFrame([
    {'label': 'Ni', 'x_Al': 0.0, 'formation_energy_per_atom_eV': 0.0},
    {'label': 'Al', 'x_Al': 1.0, 'formation_energy_per_atom_eV': 0.0},
])

# --- fcc-SQS -----------------------------------------------------------------
sqs = pd.read_csv(os.path.join(BASE, '..', 'niall_ext', 'analysis', 'niall_fcc_sqs.csv'))
sqs['Ef'] = (sqs.energy_eV / sqs.n_atoms) - ((1 - sqs.x_Al) * mu_Ni + sqs.x_Al * mu_Al)

# --- B2 off-stoichiometry branches ------------------------------------------
b2_files = [
    os.path.join(AN, 'b2_offstoich_volumes.csv'),
    os.path.join(AN, 'b2_offstoich_volumes_extra_vac.csv'),
    os.path.join(AN, 'b2_offstoich_volumes_wide_vac.csv'),
    os.path.join(AN, 'b2_offstoich_volumes_antisite_extra.csv'),
    os.path.join(AN, 'b2_offstoich_volumes_50competition.csv'),
    os.path.join(AN, 'b2_offstoich_volumes_antisite_alrich_dense.csv'),
]
b2 = pd.concat([pd.read_csv(f) for f in b2_files if os.path.exists(f)],
               ignore_index=True)
# keep converged structures; perfect always kept
b2 = b2[b2.converged | (b2.branch == 'perfect')].copy()

# aggregate per branch/composition
branch_agg = (b2[b2.branch != 'perfect']
              .groupby(['x_Al_target', 'branch'])
              .agg(x_Al=('x_Al', 'mean'),
                   Ef=('E_form_eV_atom', 'mean'),
                   Efstd=('E_form_eV_atom', 'std'),
                   a=('a_eff_A', 'mean'),
                   V=('V_per_atom_A3', 'mean'))
              .reset_index())

# add perfect B2
perfect = b2[b2.branch == 'perfect'].iloc[0]
branch_agg = pd.concat([
    branch_agg,
    pd.DataFrame([{'x_Al_target': perfect.x_Al_target, 'branch': 'perfect',
                   'x_Al': perfect.x_Al, 'Ef': perfect.E_form_eV_atom,
                   'Efstd': 0.0, 'a': perfect.a_eff_A,
                   'V': perfect.V_per_atom_A3}])
], ignore_index=True)

# --- convex hull (MACE points) ----------------------------------------------
points = []
# pure elements
points.append((0.0, 0.0))
points.append((1.0, 0.0))
# fcc-SQS individual structures (all seeds)
for _, r in sqs.iterrows():
    points.append((r.x_Al, r.Ef))
# B2 individual points
for _, r in b2.iterrows():
    if r.branch == 'perfect':
        continue
    points.append((r.x_Al, r.E_form_eV_atom))
# compounds
for _, r in compounds.iterrows():
    points.append((r.x_Al, r.formation_energy_per_atom_eV))

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
xh = pts[chain, 0]
yh = pts[chain, 1]

def hull_E(x):
    # lower hull energy by linear interpolation on chain
    return np.interp(x, xh, yh)

# --- compute delta above hull for every B2 branch mean -----------------------
branch_agg['Ef_hull'] = np.interp(branch_agg.x_Al, xh, yh)
branch_agg['delta_above_hull'] = branch_agg.Ef - branch_agg.Ef_hull

# write aggregated data with hull
branch_agg.to_csv(os.path.join(AN, 'b2_branch_hull_xmax.csv'), index=False)
print('Wrote', os.path.join(AN, 'b2_branch_hull_xmax.csv'))

# --- extract x_max ----------------------------------------------------------
TOL = 0.005  # eV/atom: consider a point "on the hull" within this band

def xmax_of_branch(br):
    g = branch_agg[branch_agg.branch == br].sort_values('x_Al')
    if g.empty: return np.nan
    on_hull = g[g.delta_above_hull <= TOL]
    if on_hull.empty: return np.nan
    return float(on_hull.x_Al.max())

xmax = {
    'antisite': xmax_of_branch('antisite'),
    'vacancy': xmax_of_branch('vacancy'),
    'experimental_B2_uniform': 0.60,
}
with open(os.path.join(AN, 'b2_xmax.json'), 'w') as f:
    json.dump(xmax, f, indent=2)
print(json.dumps(xmax, indent=2))

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(xh, yh, 'k--', lw=2.2, label='0 K 凸包（L1$_2$/Ni$_3$Al$_4$/Ni$_5$Al$_3$/Ni$_2$Al$_3$/NiAl$_3$ 含む）')

colors = {'antisite': 'tab:blue', 'vacancy': 'tab:red', 'perfect': 'tab:orange'}
labels = {'antisite': 'B2 反サイト', 'vacancy': 'B2 空孔', 'perfect': '完全 B2-NiAl'}
for br, g in branch_agg.groupby('branch'):
    g = g.sort_values('x_Al')
    ax.errorbar(g.x_Al, g.Ef, yerr=g.Efstd, fmt='o-', ms=8, capsize=3,
                color=colors.get(br, 'gray'), label=labels.get(br, br), zorder=4)

# intermetallics
for _, r in compounds.iterrows():
    ax.plot([r.x_Al], [r.formation_energy_per_atom_eV], 'D', ms=10,
            color='tab:green', zorder=6)
    ax.annotate(r.label.replace('L12_', 'L1$_2$-').replace('Ni3Al4', r'Ni$_3$Al$_4$'),
                xy=(r.x_Al, r.formation_energy_per_atom_eV),
                xytext=(0, 10), textcoords='offset points', fontsize=12, ha='center')

ax.set_xlabel(r"Al 原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"形成エネルギー $E_f$ (eV/atom)")
ax.set_title(r"B2 枝の 0 K 凸包からの乖離（$x_{\max}$ 抽出）")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.75, 0.05)
ax.legend(fontsize=12, loc='lower right')
plt.tight_layout()
outfig = os.path.join(FIG, 'fig_b2_hull_xmax.png')
plt.savefig(outfig, dpi=150)
plt.close()
print('Wrote', outfig)
