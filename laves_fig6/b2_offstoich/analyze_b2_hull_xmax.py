#!/usr/bin/env python3
"""B2 branch convex-hull analysis with finite-temperature x_max extraction.

Constructs several reference hulls:
  * 0 K hull      : all low-temperature intermetallics (Ni3Al4, Ni5Al3,
                    Ni2Al3, NiAl3, L12_Ni3Al, B2_NiAl) plus fcc-SQS.
  * 1273 K hull   : solid phases stable near 1000 C:
                    (Ni, Ni3Al, B2_NiAl, Ni2Al3).
                    Ni3Al4 and Ni5Al3 are low-temperature phases;
                    NiAl3 decomposes well below 1273 K.
  * 1473 K hull   : only Ni, Ni3Al, B2_NiAl are stable solids;
                    the Al-rich boundary is liquid, so this hull is kept
                    only as a reference and flagged as physically invalid
                    for x_max claims.

For each B2 branch point at temperature T:

    G_T(x) = E_f(x) - k_B T ln(g) / n_atoms

with g = C(64, n_defect) on the relevant sublattice of a 4x4x4 B2 supercell.

Outputs:
  analysis/b2_branch_finiteT_hull.csv : branch free energies and hull deviations
  analysis/b2_xmax.json               : x_max for several tolerance bands
  analysis/b2_xmax_sensitivity.csv    : detailed x_max sensitivity table
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

KB_EV = 8.617333262e-5
T_K_LIST = [1273.0, 1473.0]
NCELL = 64    # 4x4x4 B2 conventional cell: 64 Ni-sites + 64 Al-sites
N_SITES = 2 * NCELL

# --- phase sets -------------------------------------------------------------
PHASE_0K = ['L12_Ni3Al', 'Ni3Al4', 'Ni5Al3', 'Ni2Al3', 'NiAl3', 'B2_NiAl']
PHASE_1273 = ['L12_Ni3Al', 'Ni2Al3', 'B2_NiAl']
PHASE_1473 = ['L12_Ni3Al', 'B2_NiAl']

# --- load references ---------------------------------------------------------
mace_ref = pd.read_csv(os.path.join(AN, 'mace_mp_ref_results.csv'))
mace_ref['x_Al'] = mace_ref.n_Al / mace_ref.n_atoms
mu_Ni = mace_ref[mace_ref.label == 'Ni'].energy_per_atom_eV.values[0]
mu_Al = mace_ref[mace_ref.label == 'Al'].energy_per_atom_eV.values[0]


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
b2['ln_g_per_atom'] = b2['n_defect'].apply(lambda g: ln_comb(NCELL, g)) / b2['n_atoms']
for T in T_K_LIST:
    b2[f'G_{T:.0f}K_atom_eV'] = b2['E_form_eV_atom'] - KB_EV * T * b2['ln_g_per_atom']


# --- aggregate per target composition / branch -------------------------------
def aggregate_branches(df):
    rows = []
    for (xt, br), g in df.groupby(['x_Al_target', 'branch']):
        row = {
            'x_Al_target': xt,
            'branch': br,
            'x_Al': g.x_Al.mean(),
            'Ef': g.E_form_eV_atom.mean(),
            'Efstd': g.E_form_eV_atom.std(ddof=1) if len(g) > 1 else 0.0,
            'a': g.a_eff_A.mean(),
            'V': g.V_per_atom_A3.mean(),
            'n': len(g),
            'n_atoms': int(round(g.n_atoms.mean())),
        }
        for T in T_K_LIST:
            gcol = f'G_{T:.0f}K_atom_eV'
            gT = g[gcol]
            row[f'G_{T:.0f}K'] = gT.mean()
            row[f'Gstd_{T:.0f}K'] = gT.std(ddof=1) if len(gT) > 1 else 0.0
            row[f'minus_T_S_{T:.0f}K'] = (gT - g['E_form_eV_atom']).mean()
        rows.append(row)
    return pd.DataFrame(rows)


branch_agg = aggregate_branches(b2[b2.branch != 'perfect'])

# add perfect B2
perfect = b2[b2.branch == 'perfect'].iloc[0]
p_row = {
    'x_Al_target': perfect.x_Al_target,
    'branch': 'perfect',
    'x_Al': perfect.x_Al,
    'Ef': perfect.E_form_eV_atom,
    'Efstd': 0.0,
    'a': perfect.a_eff_A,
    'V': perfect.V_per_atom_A3,
    'n': 1,
    'n_atoms': int(perfect.n_atoms),
}
for T in T_K_LIST:
    p_row[f'G_{T:.0f}K'] = perfect.E_form_eV_atom
    p_row[f'Gstd_{T:.0f}K'] = 0.0
    p_row[f'minus_T_S_{T:.0f}K'] = 0.0
branch_agg = pd.concat([branch_agg, pd.DataFrame([p_row])], ignore_index=True)

# overall branch sampling limits (used for branch-aware saturation)
branch_max_map = branch_agg.groupby('branch')['x_Al'].max().to_dict()


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
                    ok = False
                    break
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
xh_0K, yh_0K, _ = lower_hull(build_ref_points(PHASE_0K, include_sqs=True))
xh_1273, yh_1273, _ = lower_hull(build_ref_points(PHASE_1273, include_sqs=False))
xh_1473, yh_1473, _ = lower_hull(build_ref_points(PHASE_1473, include_sqs=False))

HULLS = {
    '0K_all': (xh_0K, yh_0K),
    '1273K': (xh_1273, yh_1273),
    '1473K': (xh_1473, yh_1473),
}


def delta_to_hull(x, y, hull_x, hull_y):
    return y - np.interp(x, hull_x, hull_y)


# --- per-branch deviations ---------------------------------------------------
for label, (hx, hy) in HULLS.items():
    branch_agg[f'delta_Ef_{label}_eV'] = branch_agg.apply(
        lambda r: delta_to_hull(r.x_Al, r.Ef, hx, hy), axis=1)
    for T in T_K_LIST:
        branch_agg[f'delta_G_{T:.0f}K_{label}_eV'] = branch_agg.apply(
            lambda r, T=T, hx=hx, hy=hy: delta_to_hull(r.x_Al, r[f'G_{T:.0f}K'], hx, hy), axis=1)


# --- lower B2 branch at each temperature -------------------------------------
best_0K = branch_agg.loc[branch_agg.groupby('x_Al_target')['Ef'].idxmin()].copy()
best_0K = best_0K.sort_values('x_Al')

best_T = {}
for T in T_K_LIST:
    gcol = f'G_{T:.0f}K'
    bt = branch_agg.loc[branch_agg.groupby('x_Al_target')[gcol].idxmin()].copy()
    best_T[T] = bt.sort_values('x_Al')


def make_physical_branch(df, col):
    """Select the physically appropriate defect branch by side of stoichiometry.

    Ni-rich (x_Al < 0.5): antisite (Ni on Al sublattice).
    Stoichiometry (x_Al == 0.5): perfect B2.
    Al-rich (x_Al > 0.5): vacancy (Ni vacancies).
    """
    rows = []
    for xt, g in df.groupby('x_Al_target'):
        if abs(xt - 0.5) < 1e-9:
            allowed = ['perfect', 'vacancy', 'antisite']
        elif xt < 0.5:
            allowed = ['antisite']
        else:
            allowed = ['vacancy']
        g2 = g[g.branch.isin(allowed)].copy()
        if g2.empty:
            continue
        idx = g2[col].idxmin()
        rows.append(df.loc[idx])
    if not rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(rows).sort_values('x_Al')


best_phys_0K = make_physical_branch(branch_agg, 'Ef')
best_phys_T = {}
for T in T_K_LIST:
    best_phys_T[T] = make_physical_branch(branch_agg, f'G_{T:.0f}K')


# --- x_max extraction for several tolerances ---------------------------------
def xmax_info(df, col, tol, hull_x, hull_y, branch_col=None, branch_max_map=None):
    """Return x_max and saturation info.

    If the largest on-hull composition equals the largest sampled composition
    *of the same branch* (using the complete branch sampling range), the phase
    boundary was not crossed within the sampled range and the result is a lower
    bound (saturated=True, x_max=NaN).

    branch_max_map must be supplied for branch-aware saturation; it is built
    once from the full branch_agg so that the true per-branch sampling range is
    used regardless of which subset ``df`` is passed for the on-hull search.
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
    if on.empty:
        return {'x_max': np.nan, 'saturated': False,
                'max_sampled_x': float(g.x_Al.max()), 'n_on_hull': 0}
    max_x_on = float(on.x_Al.max())
    if branch_col is not None and branch_col in on.columns:
        if branch_max_map is None:
            raise ValueError("branch_max_map is required for branch_col saturation")
        rows_at_max = on[on.x_Al == max_x_on]
        br_at_max = rows_at_max[branch_col].iloc[0]
        branch_max_x = float(branch_max_map.get(br_at_max, max_x_on))
        saturated = abs(max_x_on - branch_max_x) < 1e-6
        max_sampled_x = branch_max_x
    else:
        saturated = abs(max_x_on - float(g.x_Al.max())) < 1e-6
        max_sampled_x = float(g.x_Al.max())
    if saturated:
        return {'x_max': np.nan, 'saturated': True, 'max_sampled_x': max_sampled_x,
                'n_on_hull': len(on)}
    else:
        return {'x_max': max_x_on, 'saturated': False, 'max_sampled_x': max_sampled_x,
                'n_on_hull': len(on)}


def do_xmax(col, branch, tol, hull_x, hull_y, best_df, temperature):
    if branch == 'best_B2_branch':
        df = best_df[['x_Al', col, 'branch']]
    else:
        df = branch_agg[branch_agg.branch == branch][['x_Al', col, 'branch']]
    info = xmax_info(df, col, tol, hull_x, hull_y,
                     branch_col='branch', branch_max_map=branch_max_map)
    return {
        'temperature': temperature,
        'tolerance_meV': int(round(tol * 1000)),
        'branch': branch,
        'energy_col': col,
        **info,
    }


tols = [0.003, 0.005, 0.010, 0.020]
xmax_records = []

# 0 K (no configurational entropy)
for tol in tols:
    for br in ['antisite', 'vacancy', 'best_B2_branch']:
        best_df = best_phys_0K if br == 'best_B2_branch' else best_0K
        xmax_records.append(do_xmax('Ef', br, tol, xh_0K, yh_0K, best_df, '0K_all'))

# finite temperatures
for T in T_K_LIST:
    gcol = f'G_{T:.0f}K'
    hx, hy = HULLS[f'{T:.0f}K']
    for tol in tols:
        for br in ['antisite', 'vacancy', 'best_B2_branch']:
            best_df = best_phys_T[T] if br == 'best_B2_branch' else best_T[T]
            xmax_records.append(do_xmax(gcol, br, tol, hx, hy, best_df, f'{T:.0f}K'))

xmax_df = pd.DataFrame(xmax_records)


# --- JSON summary ------------------------------------------------------------
exp_xmax = {
    '0K_all': 0.60,    # experimental single-phase limit at 0 K is not well defined
    '1273K': 0.575,    # NiAl Al-rich solid limit at ~1000 C is ~57-58 at.% Al
    '1473K': np.nan,   # liquid boundary, not a solid-solid comparison
}

oxmax = {}
for T_key, ecol, best_df in [('0K_all', 'Ef', best_phys_0K),
                             ('1273K', 'G_1273K', best_phys_T[1273.0]),
                             ('1473K', 'G_1473K', best_phys_T[1473.0])]:
    oxmax[T_key] = {}
    for tol in tols:
        sub = xmax_df[
            (xmax_df.temperature == T_key) &
            (xmax_df.tolerance_meV == int(round(tol * 1000))) &
            (xmax_df.branch == 'best_B2_branch') &
            (xmax_df.energy_col == ecol)
        ]
        if not sub.empty:
            row = sub.iloc[0]
            entry = {
                'x_max': None if pd.isna(row['x_max']) else round(float(row['x_max']), 4),
                'saturated': bool(row['saturated']),
            }
            if bool(row['saturated']):
                entry['lower_bound_x'] = round(float(row['max_sampled_x']), 4)
            else:
                entry['max_sampled_x'] = round(float(row['max_sampled_x']), 4)
            oxmax[T_key][f"tol_{int(round(tol*1000))}meV"] = entry
    oxmax[T_key]['experimental_B2_single_phase_approx'] = exp_xmax[T_key]

oxmax['temperature_K'] = T_K_LIST[0]  # primary comparison temperature
oxmax['note'] = """1273 K uses only solid competitors (Ni, Ni3Al, B2, Ni2Al3).
1473 K is above Ni2Al3 and NiAl3 stability and the NiAl Al-rich boundary is
liquid; the 1473 K results are therefore shown for reference only and are not
a valid solid-state x_max comparison."""

with open(os.path.join(AN, 'b2_xmax.json'), 'w') as f:
    json.dump(oxmax, f, indent=2, default=float)
print(json.dumps(oxmax, indent=2, default=float))

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
ax.plot(xh_0K, yh_0K, 'k--', lw=2.0,
        label='0 K 凸包（Ni$_3$Al$_4$/Ni$_5$Al$_3$/Ni$_2$Al$_3$/NiAl$_3$ 含む）')
ax.plot(xh_1273, yh_1273, 'k-', lw=2.4,
        label='1273 K 凸包（Ni$_3$Al$_4$/Ni$_5$Al$_3$/NiAl$_3$ 除く）')
ax.plot(xh_1473, yh_1473, 'k:', lw=1.8,
        label='1473 K 参考（Al-rich は液相、固相のみ）')

colors = {'antisite': 'tab:blue', 'vacancy': 'tab:red', 'perfect': 'tab:orange'}
labels = {'antisite': 'B2 反サイト', 'vacancy': 'B2 空孔', 'perfect': '完全 B2-NiAl'}

# 0 K Ef (open)
for br, g in branch_agg[branch_agg.branch != 'perfect'].groupby('branch'):
    g = g.sort_values('x_Al')
    ax.errorbar(g.x_Al, g.Ef, yerr=g.Efstd, fmt='o--', ms=7, capsize=3,
                mfc='none', color=colors[br], label=f'{labels[br]} $E_f$ (0 K)', zorder=4)

# 1273 K G (filled)
for br, g in branch_agg[branch_agg.branch != 'perfect'].groupby('branch'):
    g = g.sort_values('x_Al')
    ax.errorbar(g.x_Al, g['G_1273K'], yerr=g['Gstd_1273K'], fmt='o-', ms=8, capsize=3,
                color=colors[br], label=f'{labels[br]} $G$ (1273 K, -$TS_{{conf}}$)', zorder=5)

# perfect B2
p = branch_agg[branch_agg.branch == 'perfect'].iloc[0]
ax.plot([p.x_Al], [p.Ef], 's', ms=12, color=colors['perfect'], label=labels['perfect'], zorder=6)

# intermetallics (0 K set)
comp_all = get_compounds(PHASE_0K)
for _, r in comp_all.iterrows():
    ax.plot([r.x_Al], [r.formation_energy_per_atom_eV], 'D', ms=9,
            color='tab:green', zorder=6)
    label = r.label.replace('L12_', 'L1$_2$-')
    for s, repl in [('Ni3Al4', 'Ni$_3$Al$_4$'), ('Ni5Al3', 'Ni$_5$Al$_3$'),
                    ('Ni2Al3', 'Ni$_2$Al$_3$'), ('NiAl3', 'NiAl$_3$')]:
        label = label.replace(s, repl)
    ax.annotate(label, xy=(r.x_Al, r.formation_energy_per_atom_eV),
                textcoords='offset points', xytext=(6, 6), fontsize=12,
                color='darkgreen')

# x_max markers for 1273 K
sub1273 = xmax_df[(xmax_df.temperature == '1273K') &
                  (xmax_df.tolerance_meV == 5) &
                  (xmax_df.branch == 'best_B2_branch')]
if not sub1273.empty:
    row = sub1273.iloc[0]
    if pd.isna(row['x_max']):
        xmark = row['max_sampled_x']
        ax.axvline(xmark, color='tab:purple', ls='-.', lw=2,
                   label=f"1273 K $x_{{\max}}$ ≳ {xmark:.3f} (saturated, lower bound)")
    else:
        ax.axvline(row['x_max'], color='tab:purple', ls='-.', lw=2,
                   label=f"1273 K $x_{{\max}}$ = {row['x_max']:.3f}")

ax.axhline(0.0, color='gray', lw=0.8)
ax.set_xlabel('$x_{\\rm Al}$', fontsize=18)
ax.set_ylabel('形成エネルギー / Helmholtz 自由エネルギー (eV/atom)', fontsize=18)
ax.set_title('B2-NiAl 不定比モデル vs 0 K / 1273 K 凸包', fontsize=20)
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_b2_hull_finiteT.png'), dpi=150)
plt.close()
print('Wrote', os.path.join(FIG, 'fig_b2_hull_finiteT.png'))


# --- fig: 0 K hull with B2 defect models (x_max extraction helper) ----------
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(xh_0K, yh_0K, 'k--', lw=2.2,
         label='0 K 凸包（L1$_2$/Ni$_3$Al$_4$/Ni$_5$Al$_3$/Ni$_2$Al$_3$/NiAl$_3$ 含む）')

colors_x = {'antisite': 'tab:blue', 'vacancy': 'tab:red', 'perfect': 'tab:orange'}
labels_x = {'antisite': 'B2 反サイト', 'vacancy': 'B2 空孔', 'perfect': '完全 B2-NiAl'}
for br, g in branch_agg.groupby('branch'):
    g = g.sort_values('x_Al')
    ax2.errorbar(g.x_Al, g.Ef, yerr=g.Efstd, fmt='o-', ms=8, capsize=3,
                 color=colors_x.get(br, 'gray'),
                 label=labels_x.get(br, br), zorder=4)

for _, r in comp_all.iterrows():
    ax2.plot([r.x_Al], [r.formation_energy_per_atom_eV], 'D', ms=10,
             color='tab:green', zorder=6)
    label = r.label.replace('L12_', 'L1$_2$-')
    for s, repl in [('Ni3Al4', 'Ni$_3$Al$_4$'), ('Ni5Al3', 'Ni$_5$Al$_3$'),
                    ('Ni2Al3', 'Ni$_2$Al$_3$'), ('NiAl3', 'NiAl$_3$')]:
        label = label.replace(s, repl)
    ax2.annotate(label, xy=(r.x_Al, r.formation_energy_per_atom_eV),
                 textcoords='offset points', xytext=(0, 10), fontsize=12,
                 ha='center', color='darkgreen')

ax2.set_xlabel(r"$x_{\mathrm{Al}}$")
ax2.set_ylabel(r"形成エネルギー $E_f$ (eV/atom)")
ax2.set_title(r"B2 欠陥モデルの 0 K 凸包からの乖離（$x_{\max}$ 抽出）", fontsize=18)
ax2.set_xlim(-0.03, 1.03)
ax2.set_ylim(-0.75, 0.05)
ax2.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_b2_hull_xmax.png'), dpi=150)
plt.close()
print('Wrote', os.path.join(FIG, 'fig_b2_hull_xmax.png'))
