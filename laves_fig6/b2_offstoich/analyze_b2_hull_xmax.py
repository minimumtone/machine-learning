#!/usr/bin/env python3
"""B2 branch convex-hull analysis with finite-temperature x_max extraction.

Constructs two reference hulls:
  * 0 K hull      : all low-temperature intermetallics (Ni3Al4, Ni5Al3,
                    Ni2Al3, NiAl3, L12_Ni3Al, B2_NiAl) plus fcc-SQS.
  * 1473 K hull   : high-temperature phase set
                    (Ni, Ni3Al, B2_NiAl, Ni2Al3, NiAl3 only;
                     Ni3Al4 and Ni5Al3 are reported only below ~700 degC).

For each B2 branch point the Helmholtz free energy per occupied atom is

    G(x) = E_f(x) - k_B T  ln(g) / n_atoms

with g = C(64, n_defect) on the relevant sublattice of a 4x4x4 B2 supercell.
The finite-temperature B2 homogeneous limit is the largest x where the
lower B2 branch curve lies below the 1473 K reference hull after the
configurational entropy correction is applied.

Outputs:
  analysis/b2_branch_finiteT_hull.csv : branch free energies, -T*S_conf,
                                         and deviations from both hulls
  analysis/b2_xmax.json               : x_max for several tolerance bands
  figures/fig_b2_hull_finiteT.png     : B2 branch free energies vs hulls
"""
import json, math, os
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

T_K = 1473.0
KT_EV = 8.617333262e-5 * T_K
NCELL = 64    # 4x4x4 B2 conventional cell: 64 Ni-sites + 64 Al-sites
N_SITES = 2 * NCELL

# --- load references ---------------------------------------------------------
mace_ref = pd.read_csv(os.path.join(AN, 'mace_mp_ref_results.csv'))
mace_ref['x_Al'] = mace_ref.n_Al / mace_ref.n_atoms
mu_Ni = mace_ref[mace_ref.label == 'Ni'].energy_per_atom_eV.values[0]
mu_Al = mace_ref[mace_ref.label == 'Al'].energy_per_atom_eV.values[0]

compounds_all = ['L12_Ni3Al', 'Ni3Al4', 'Ni5Al3', 'Ni2Al3', 'NiAl3', 'B2_NiAl']
compounds_1473 = ['L12_Ni3Al', 'Ni2Al3', 'NiAl3', 'B2_NiAl']

def get_compounds(names):
    return mace_ref[mace_ref.label.isin(names)].copy()

# pure elements
pure = pd.DataFrame([
    {'label': 'Ni', 'x_Al': 0.0, 'formation_energy_per_atom_eV': 0.0},
    {'label': 'Al', 'x_Al': 1.0, 'formation_energy_per_atom_eV': 0.0},
])

# fcc-SQS (0 K only)
sqs_path = os.path.join(BASE, '..', 'niall_ext', 'analysis', 'niall_fcc_sqs.csv')
if os.path.exists(sqs_path):
    sqs = pd.read_csv(sqs_path)
    sqs['Ef'] = (sqs.energy_eV / sqs.n_atoms) - ((1 - sqs.x_Al) * mu_Ni + sqs.x_Al * mu_Al)
else:
    sqs = pd.DataFrame({'x_Al': [], 'Ef': []})

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
b2 = b2[b2.converged | (b2.branch == 'perfect')].copy()

# --- configurational entropy on the point-defect sublattice ------------------
def ln_comb(n, k):
    if k <= 0 or k >= n or n < 0:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def n_defect(row):
    if row.branch == 'perfect':
        return 0
    if row.branch == 'antisite':
        if row.x_Al_target < 0.5:
            return int(round(row.n_Ni - NCELL))       # Ni on Al sublattice
        else:
            return int(round(row.n_Al - NCELL))       # Al on Ni sublattice
    elif row.branch == 'vacancy':
        if row.x_Al_target < 0.5:
            return int(round(NCELL - row.n_Al))       # vacancies on Al sublattice
        else:
            return int(round(NCELL - row.n_Ni))       # vacancies on Ni sublattice
    return 0


b2['n_defect'] = b2.apply(n_defect, axis=1)
b2['minus_T_S_per_atom_eV'] = -KT_EV * b2['n_defect'].apply(lambda g: ln_comb(NCELL, g)) / b2['n_atoms']
b2['G_atom_eV'] = b2['E_form_eV_atom'] + b2['minus_T_S_per_atom_eV']

# --- aggregate per target composition / branch -------------------------------
def group_stats(g):
    return {'x_Al': g.x_Al.mean(),
            'Ef': g.E_form_eV_atom.mean(),
            'Efstd': g.E_form_eV_atom.std(ddof=1) if len(g) > 1 else 0.0,
            'minus_T_S': (g.minus_T_S_per_atom_eV).mean(),
            'G': g.G_atom_eV.mean(),
            'Gstd': g.G_atom_eV.std(ddof=1) if len(g) > 1 else 0.0,
            'a': g.a_eff_A.mean(),
            'V': g.V_per_atom_A3.mean(),
            'n': len(g),
            'n_atoms': int(round(g.n_atoms.mean()))}

branch_agg = (b2[b2.branch != 'perfect']
              .groupby(['x_Al_target', 'branch'])
              .apply(group_stats)
              .apply(pd.Series)
              .reset_index())

# add perfect B2
perfect = b2[b2.branch == 'perfect'].iloc[0]
branch_agg = pd.concat([
    branch_agg,
    pd.DataFrame([{'x_Al_target': perfect.x_Al_target, 'branch': 'perfect',
                   'x_Al': perfect.x_Al, 'Ef': perfect.E_form_eV_atom,
                   'Efstd': 0.0, 'minus_T_S': 0.0, 'G': perfect.E_form_eV_atom,
                   'Gstd': 0.0, 'a': perfect.a_eff_A, 'V': perfect.V_per_atom_A3,
                   'n': 1, 'n_atoms': int(perfect.n_atoms)}])
], ignore_index=True)

# --- helper: lower convex hull as x-sorted chain -----------------------------
def lower_hull(points):
    pts = np.array(points)
    if len(pts) <= 1:
        idx = np.argsort(pts[:, 0])
        return pts[idx, 0], pts[idx, 1], idx
    hull = ConvexHull(pts)
    edges = []
    for i, j in hull.simplices:
        x1, y1 = pts[i]; x2, y2 = pts[j]
        if x1 > x2:
            i, j = j, i; x1, y1 = pts[i]; x2, y2 = pts[j]
        if abs(x2 - x1) < 1e-9:
            continue
        m = (y2 - y1) / (x2 - x1)
        ok = True
        for k in range(len(pts)):
            if k == i or k == j:
                continue
            xx, yy = pts[k]
            if min(x1, x2) - 1e-9 <= xx <= max(x1, x2) + 1e-9:
                if yy < m * (xx - x1) + y1 - 1e-7:
                    ok = False; break
        if ok:
            edges.append((i, j))
    adj = {}
    for i, j in edges:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    start = int(np.argmin(pts[:, 0]))
    seen = set(); chain = []
    def walk(i):
        seen.add(i); chain.append(i)
        nx = [j for j in adj.get(i, []) if j not in seen and pts[j, 0] >= pts[i, 0] - 1e-9]
        if nx:
            walk(min(nx, key=lambda k: pts[k, 0]))
    walk(start)
    chain = sorted(chain, key=lambda i: pts[i, 0])
    return pts[chain, 0], pts[chain, 1], chain


def build_ref_points(compound_labels, include_sqs=False):
    pts = [(0.0, 0.0), (1.0, 0.0)]
    comp = get_compounds(compound_labels)
    for _, r in comp.iterrows():
        pts.append((r.x_Al, r.formation_energy_per_atom_eV))
    if include_sqs and not sqs.empty:
        for _, r in sqs.iterrows():
            pts.append((r.x_Al, r.Ef))
    return pts

# --- reference hulls ---------------------------------------------------------
pts_0K = build_ref_points(compounds_all, include_sqs=True)
xh_0K, yh_0K, _ = lower_hull(pts_0K)

pts_1473 = build_ref_points(compounds_1473, include_sqs=False)
xh_1473, yh_1473, _ = lower_hull(pts_1473)

# --- per-branch deviations ---------------------------------------------------
def delta_to_hull(x, y, hull_x, hull_y):
    return y - np.interp(x, hull_x, hull_y)

branch_agg['Ef_minus_0K_hull_eV'] = branch_agg.apply(
    lambda r: delta_to_hull(r.x_Al, r.Ef, xh_0K, yh_0K), axis=1)
branch_agg['G_minus_0K_hull_eV'] = branch_agg.apply(
    lambda r: delta_to_hull(r.x_Al, r.G, xh_0K, yh_0K), axis=1)
branch_agg['G_minus_1473_hull_eV'] = branch_agg.apply(
    lambda r: delta_to_hull(r.x_Al, r.G, xh_1473, yh_1473), axis=1)

# lower B2 branch at each composition
best = branch_agg.loc[branch_agg.groupby('x_Al_target')['G'].idxmin()].copy()
best = best.sort_values('x_Al')

# --- x_max extraction for several tolerances ---------------------------------
def xmax_info(df, col, tol, hull_x, hull_y, branch_col=None):
    """Return x_max and saturation info.

    If the largest on-hull composition equals the largest sampled composition
    *of the same branch*, the phase boundary was NOT crossed within the sampled
    range for that branch; x_max is returned as NaN and saturated=True with a
    lower-bound value.  This avoids the false positive where another branch has
    sampled points beyond the branch that determines the boundary.
    """
    cols = ['x_Al', col]
    if branch_col is not None:
        cols.append(branch_col)
    g = df[cols].copy().sort_values('x_Al')
    if g.empty:
        return {'x_max': np.nan, 'saturated': False, 'max_sampled_x': np.nan,
                'n_on_hull': 0}
    delta = g[col] - np.interp(g.x_Al.values, hull_x, hull_y)
    on = g[delta <= tol]
    max_x_all = float(g.x_Al.max())
    if on.empty:
        return {'x_max': np.nan, 'saturated': False, 'max_sampled_x': max_x_all,
                'n_on_hull': 0}
    max_x_on = float(on.x_Al.max())
    if branch_col is not None and branch_col in on.columns:
        # Use the branch of the point at max_x_on instead of the global maximum.
        rows_at_max = on[on.x_Al == max_x_on]
        br_at_max = rows_at_max[branch_col].iloc[0]
        branch_max_x = float(g[g[branch_col] == br_at_max].x_Al.max())
        saturated = abs(max_x_on - branch_max_x) < 1e-6
        max_sampled_x = branch_max_x
    else:
        saturated = abs(max_x_on - max_x_all) < 1e-6
        max_sampled_x = max_x_all
    if saturated:
        return {'x_max': np.nan, 'saturated': True, 'max_sampled_x': max_sampled_x,
                'n_on_hull': len(on)}
    else:
        return {'x_max': max_x_on, 'saturated': False, 'max_sampled_x': max_sampled_x,
                'n_on_hull': len(on)}


def xmax_from_branch(br, col, tol, hull_x, hull_y):
    return xmax_info(branch_agg[branch_agg.branch == br], col, tol, hull_x, hull_y,
                     branch_col='branch')


def xmax_overall(col, tol, hull_x, hull_y):
    return xmax_info(best[['x_Al', col, 'branch']], col, tol, hull_x, hull_y,
                     branch_col='branch')


def record(temperature, branch, energy_col, tol, hull_x, hull_y):
    if branch == 'best_B2_branch':
        info = xmax_overall(energy_col, tol, hull_x, hull_y)
    else:
        info = xmax_from_branch(branch, energy_col, tol, hull_x, hull_y)
    return {
        'temperature': temperature,
        'tolerance_meV': round(tol * 1000),
        'branch': branch,
        'energy_col': energy_col,
        **info,
    }


tols = [0.003, 0.005, 0.010, 0.020]
xmax_records = []
for tol in tols:
    for br in ['antisite', 'vacancy']:
        # 0 K: formation energy (Ef) and Helmholtz free energy (G)
        xmax_records.append(record('0K_all', br, 'Ef', tol, xh_0K, yh_0K))
        xmax_records.append(record('0K_all', br, 'G', tol, xh_0K, yh_0K))
        # 1473 K: only G is physical for the high-T comparison
        xmax_records.append(record('1473K', br, 'G', tol, xh_1473, yh_1473))
    xmax_records.append(record('0K_all', 'best_B2_branch', 'Ef', tol, xh_0K, yh_0K))
    xmax_records.append(record('0K_all', 'best_B2_branch', 'G', tol, xh_0K, yh_0K))
    xmax_records.append(record('1473K', 'best_B2_branch', 'G', tol, xh_1473, yh_1473))

xmax_df = pd.DataFrame(xmax_records)

# JSON summary: primary x_max metric vs physical energy column
oxmax = {}
for T, ecol in [('0K_all', 'Ef'), ('1473K', 'G')]:
    oxmax[T] = {}
    for tol in tols:
        sub = xmax_df[(xmax_df.temperature == T) &
                      (xmax_df.tolerance_meV == round(tol * 1000)) &
                      (xmax_df.branch == 'best_B2_branch') &
                      (xmax_df.energy_col == ecol)]
        if not sub.empty:
            row = sub.iloc[0]
            entry = {
                'x_max': None if pd.isna(row['x_max']) else round(float(row['x_max']), 4),
                'saturated': bool(row['saturated']),
            }
            if bool(row['saturated']):
                entry['lower_bound_x'] = round(float(row['max_sampled_x']), 4)
            oxmax[T][f"tol_{round(tol*1000)}meV"] = entry

oxmax['experimental_B2_uniform'] = 0.60
oxmax['temperature_K'] = T_K
with open(os.path.join(AN, 'b2_xmax.json'), 'w') as f:
    json.dump(oxmax, f, indent=2)
print(json.dumps(oxmax, indent=2))

# --- save tables -------------------------------------------------------------
xmax_df.to_csv(os.path.join(AN, 'b2_xmax_sensitivity.csv'), index=False)
print('Wrote', os.path.join(AN, 'b2_xmax_sensitivity.csv'))

branch_agg = branch_agg.sort_values(['branch', 'x_Al'])
branch_agg.to_csv(os.path.join(AN, 'b2_branch_finiteT_hull.csv'), index=False)
print('Wrote', os.path.join(AN, 'b2_branch_finiteT_hull.csv'))

# --- plot -------------------------------------------------------------------
plt.rcParams.update({'font.size': 16, 'axes.grid': True, 'grid.alpha': 0.3,
                     'font.family': ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif'],
                     'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(12, 8))
# reference hulls
ax.plot(xh_0K, yh_0K, 'k--', lw=2.0, label='0 K 凸包（Ni$_3$Al$_4$/Ni$_5$Al$_3$ 含む）')
ax.plot(xh_1473, yh_1473, 'k-', lw=2.4, label='1473 K 凸包（Ni$_3$Al$_4$/Ni$_5$Al$_3$ 除く）')

colors = {'antisite': 'tab:blue', 'vacancy': 'tab:red', 'perfect': 'tab:orange'}
labels = {'antisite': 'B2 反サイト', 'vacancy': 'B2 空孔', 'perfect': '完全 B2-NiAl'}

# B2 branches: 0 K Ef (open) and G at 1473 K (filled)
for br, g in branch_agg[branch_agg.branch != 'perfect'].groupby('branch'):
    g = g.sort_values('x_Al')
    ax.errorbar(g.x_Al, g.Ef, yerr=g.Efstd, fmt='o--', ms=7, capsize=3,
                mfc='none', color=colors[br], label=f'{labels[br]} $E_f$ (0 K)', zorder=4)
    ax.errorbar(g.x_Al, g.G, yerr=g.Gstd, fmt='o-', ms=8, capsize=3,
                color=colors[br], label=f'{labels[br]} $G$ ({T_K:.0f} K, -$TS_{{conf}}$)', zorder=5)

# perfect B2
p = branch_agg[branch_agg.branch == 'perfect'].iloc[0]
ax.plot([p.x_Al], [p.G], 's', ms=12, color=colors['perfect'], label=labels['perfect'], zorder=6)

# intermetallics (0 K set)
comp_all = get_compounds(compounds_all)
for _, r in comp_all.iterrows():
    ax.plot([r.x_Al], [r.formation_energy_per_atom_eV], 'D', ms=9,
            color='tab:green', zorder=6)
    label = r.label.replace('L12_', 'L1$_2$-')
    for s, repl in [('Ni3Al4', 'Ni$_3$Al$_4$'), ('Ni5Al3', 'Ni$_5$Al$_3$'),
                    ('Ni2Al3', 'Ni$_2$Al$_3$'), ('NiAl3', 'NiAl$_3$')]:
        label = label.replace(s, repl)
    ax.annotate(label, xy=(r.x_Al, r.formation_energy_per_atom_eV),
                xytext=(0, 10), textcoords='offset points', fontsize=11, ha='center')

# vertical band for experimental homogeneous range
ax.axvspan(0.45, 0.60, color='gray', alpha=0.08, label='実験 B2 均一域（Ellner）')

ax.set_xlabel("Al 原子分率 $x_{Al}$")
ax.set_ylabel("エネルギー $E_f$ / $G$ (eV/atom)")
ax.set_title("B2 枝の 0 K・1473 K 凸包からの乖離（$-TS_{conf}$ 込み）")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.75, 0.05)
ax.legend(fontsize=10, loc='lower right', ncol=2)
plt.tight_layout()
outfig = os.path.join(FIG, 'fig_b2_hull_finiteT.png')
plt.savefig(outfig, dpi=150)
plt.close()
print('Wrote', outfig)
