#!/usr/bin/env python3
"""
Fix old-vs-new TDB comparison and generate additional analysis for report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from numpy.linalg import lstsq
import os, warnings, re
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")
TITLE_SIZE = 18; LABEL_SIZE = 15; TICK_SIZE = 13; LEGEND_SIZE = 12; ANNOT_SIZE = 11
OUT = "/tmp/fev_detailed_report"

# ============================================================
# 1. Fix Old vs New TDB GF value comparison
# ============================================================
print("="*70)
print("1. Fixing Old vs New TDB GF Comparison")
print("="*70)

def parse_gf_functions(tdb_path):
    """Parse GFxVyyy function values from TDB file correctly."""
    gf = {}
    with open(tdb_path) as f:
        content = f.read()
    # Match: FUNCTION GFxVyyy 298.15 <value>; 6000 N !
    pattern = r'FUNCTION\s+(GF\d+V\d+)\s+298\.15\s+([-\d.]+)\s*;'
    for match in re.finditer(pattern, content):
        name = match.group(1)
        val = float(match.group(2))
        gf[name] = val
    return gf

old_gf = parse_gf_functions("/home/ubuntu/repos/machine-learning/calphad/Fe-V_B2_221.tdb")
new_gf = parse_gf_functions("/home/ubuntu/repos/machine-learning/calphad/Fe-V_B2_221_VASP.tdb")

print(f"Old TDB (Book1.xlsx): {len(old_gf)} B2_221 GF functions")
print(f"New TDB (Fe-V_VASP.xlsx): {len(new_gf)} B2_221 GF functions")

common = sorted(set(old_gf.keys()) & set(new_gf.keys()))
print(f"Common function names: {len(common)}")

# Compare
old_vals = np.array([old_gf[f] for f in common])
new_vals = np.array([new_gf[f] for f in common])
diffs = new_vals - old_vals

print("\nDifferences (New - Old):")
print(f"  Mean  = {np.mean(diffs):.0f} J/mol")
print(f"  Std   = {np.std(diffs):.0f} J/mol")
print(f"  Min   = {np.min(diffs):.0f} J/mol")
print(f"  Max   = {np.max(diffs):.0f} J/mol")
print(f"  Median = {np.median(diffs):.0f} J/mol")

# Also get composition info for each GF function
gf_comp = {}
for name in common:
    nv = int(name.split('V')[0].replace('GF', ''))
    gf_comp[name] = nv

# Show some examples
print("\nSample comparisons:")
for func in common[:5]:
    print(f"  {func}: Old={old_gf[func]:.0f}, New={new_gf[func]:.0f}, Diff={new_gf[func]-old_gf[func]:.0f}")

# Functions only in old or new
old_only = set(old_gf.keys()) - set(new_gf.keys())
new_only = set(new_gf.keys()) - set(old_gf.keys())
if old_only:
    print(f"\nFunctions only in OLD TDB ({len(old_only)}): {sorted(old_only)[:10]}...")
if new_only:
    print(f"\nFunctions only in NEW TDB ({len(new_only)}): {sorted(new_only)[:10]}...")

# ============================================================
# Regenerate Figure 7 with correct values
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 7a: Parity plot
ax = axes[0, 0]
nv_colors = [gf_comp[f] for f in common]
scatter = ax.scatter(old_vals/1000, new_vals/1000, c=nv_colors, cmap='viridis',
                    s=40, alpha=0.7, edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$n_V$ (V atoms)', fontsize=LABEL_SIZE)
lim = [min(old_vals.min(), new_vals.min())/1000 - 20, max(old_vals.max(), new_vals.max())/1000 + 20]
ax.plot(lim, lim, 'r--', linewidth=1.5)
ax.set_xlabel('Old TDB (Book1.xlsx) GF (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel('New TDB (Fe-V_VASP.xlsx) GF (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(f'(a) Old vs New GF Values ({len(common)} common)', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 7b: Histogram of differences
ax = axes[0, 1]
ax.hist(diffs/1000, bins=40, color=PALETTE[0], alpha=0.7, edgecolor='gray')
ax.axvline(x=np.mean(diffs)/1000, color='red', linewidth=2, 
          label=f'Mean = {np.mean(diffs)/1000:.1f} kJ/mol')
ax.axvline(x=np.median(diffs)/1000, color='orange', linewidth=2, linestyle='--',
          label=f'Median = {np.median(diffs)/1000:.1f} kJ/mol')
ax.set_xlabel('New - Old (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel('Count', fontsize=LABEL_SIZE)
ax.set_title('(b) Distribution of GF Differences', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 7c: Difference by composition
ax = axes[1, 0]
comp_diffs = {}
for func in common:
    nv = gf_comp[func]
    if nv not in comp_diffs:
        comp_diffs[nv] = []
    comp_diffs[nv].append((new_gf[func] - old_gf[func])/1000)

bp_data = []
bp_labels = []
for nv in sorted(comp_diffs.keys()):
    bp_data.append(comp_diffs[nv])
    bp_labels.append(f'{8-nv}Fe{nv}V')
bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], sns.color_palette('viridis', len(bp_data))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Composition', fontsize=LABEL_SIZE)
ax.set_ylabel('New - Old GF (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(c) GF Differences by Composition', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.tick_params(axis='x', rotation=45)

# 7d: Summary table
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['', 'Old (Book1)', 'New (VASP)', 'Diff'],
    ['Data Source', 'Book1.xlsx', 'Fe-V_VASP.xlsx', '-'],
    ['Convergence Rate', '74.6%', '96.5%', '+21.9%'],
    ['Fe Reference', 'Wang et al.', 'DFT direct', '-'],
    ['V Reference', 'Wang et al.', 'DFT direct', '-'],
    ['GF Functions', f'{len(old_gf)}', f'{len(new_gf)}', f'{len(common)} common'],
    [r'Mean $\Delta$GF (kJ/mol)', '-', '-', f'{np.mean(diffs)/1000:.1f}'],
    [r'Std $\Delta$GF (kJ/mol)', '-', '-', f'{np.std(diffs)/1000:.1f}'],
    [r'Corr(old, new)', '-', '-', f'{np.corrcoef(old_vals, new_vals)[0,1]:.4f}'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(ANNOT_SIZE+1)
table.scale(1.3, 1.8)
for j in range(4):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('(d) Dataset Comparison Summary', fontsize=TITLE_SIZE, pad=20)

plt.tight_layout()
plt.savefig(f'{OUT}/07_old_vs_new.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Regenerated 07_old_vs_new.png")

# ============================================================
# 2. RK fit to composition AVERAGES (proper comparison)
# ============================================================
print("\n" + "="*70)
print("2. Redlich-Kister Fit to Composition Averages")
print("="*70)

df = pd.read_csv("/home/ubuntu/repos/machine-learning/calphad/fev_vasp_corrected_data.csv")

# Compute composition averages
comp_avg = df.groupby('x_V')['dH_f_Jmol_corrected'].mean().reset_index()
comp_avg.columns = ['x_V', 'mean_dHf']

# RK basis for averages (exclude pure elements)
mask = (comp_avg['x_V'] > 0) & (comp_avg['x_V'] < 1)
x_avg = comp_avg.loc[mask, 'x_V'].values
dHf_avg = comp_avg.loc[mask, 'mean_dHf'].values

def rk_basis(x_V_arr, n_terms):
    x_Fe = 1 - x_V_arr
    X = np.zeros((len(x_V_arr), n_terms))
    for v in range(n_terms):
        X[:, v] = x_Fe * x_V_arr * (x_Fe - x_V_arr)**v
    return X

print("\nRK fit to composition-averaged formation energies:")
for n in [1, 2, 3, 4]:
    X = rk_basis(x_avg, n)
    L, _, _, _ = lstsq(X, dHf_avg, rcond=None)
    pred = X @ L
    rmse = np.sqrt(np.mean((dHf_avg - pred)**2))
    r2 = 1 - np.sum((dHf_avg - pred)**2) / np.sum((dHf_avg - np.mean(dHf_avg))**2)
    terms_str = ', '.join([f'L{v}={L[v]:.0f}' for v in range(n)])
    print(f"  RK-{n}: RMSE={rmse:.0f} J/mol, R²={r2:.4f} | {terms_str}")

# Best fit (4-term) for plot
X4 = rk_basis(x_avg, 4)
L4, _, _, _ = lstsq(X4, dHf_avg, rcond=None)
pred4 = X4 @ L4

X2 = rk_basis(x_avg, 2)
L2, _, _, _ = lstsq(X2, dHf_avg, rcond=None)
pred2 = X2 @ L2

# ============================================================
# 3. Generate improved interaction parameter figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 3a: RK fit to composition averages
ax = axes[0, 0]
x_dense = np.linspace(0.001, 0.999, 500)
for n, label, color, ls in [(2, 'RK-2 (DFT avg)', PALETTE[1], '-'),
                              (4, 'RK-4 (DFT avg)', PALETTE[3], '-')]:
    Xd = rk_basis(x_dense, n)
    Xf = rk_basis(x_avg, n)
    Lf, _, _, _ = lstsq(Xf, dHf_avg, rcond=None)
    yd = Xd @ Lf
    ax.plot(x_dense, yd/1000, ls, color=color, linewidth=2, label=label)

# HKR parameters
L0_HKR, L1_HKR = -21427, 7345
y_hkr = x_dense * (1-x_dense) * (L0_HKR + L1_HKR*(1-2*x_dense))
ax.plot(x_dense, y_hkr/1000, 'k--', linewidth=2, label='HKR (1991)')

# DFT individual points
df_mix = df[(df['x_V'] > 0) & (df['x_V'] < 1)]
ax.scatter(df_mix['x_V'], df_mix['dH_f_Jmol_corrected']/1000, 
          c='lightgray', s=10, alpha=0.3, label='Individual configs')
# Composition averages
ax.scatter(x_avg, dHf_avg/1000, c='red', s=100, zorder=5,
          edgecolors='black', linewidth=1, label='DFT composition avg')
comp_std = df.groupby('x_V')['dH_f_Jmol_corrected'].std()
std_vals = comp_std.loc[x_avg].values
ax.errorbar(x_avg, dHf_avg/1000, yerr=std_vals/1000, fmt='none', ecolor='red', alpha=0.5)

ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(a) RK Fits vs DFT (Composition Averages)', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE-1, loc='upper right')
ax.tick_params(labelsize=TICK_SIZE)

# 3b: Scatter at each composition showing spread
ax = axes[0, 1]
# For each composition, show ordered vs disordered energy
comps = sorted(df['composition'].unique())
x_positions = []
e_ordered = []
e_disordered_mean = []
e_spread = []
for comp in comps:
    sub = df[df['composition'] == comp]
    nv = sub['n_v'].iloc[0]
    if nv == 0 or nv == 8:
        continue
    xv = nv/8.0
    x_positions.append(xv)
    e_ordered.append(sub['dH_f_Jmol_corrected'].min()/1000)
    e_disordered_mean.append(sub['dH_f_Jmol_corrected'].mean()/1000)
    e_spread.append(sub['dH_f_Jmol_corrected'].max()/1000 - sub['dH_f_Jmol_corrected'].min()/1000)

ax.bar(np.array(x_positions) - 0.02, np.array(e_disordered_mean) - np.array(e_ordered), 
      bottom=e_ordered, width=0.04, color=PALETTE[3], alpha=0.6, label='Ordering energy')
ax.plot(x_positions, e_ordered, 'v-', color=PALETTE[2], linewidth=2, markersize=10, 
       label='Most stable (B2-like)')
ax.plot(x_positions, e_disordered_mean, 'o-', color=PALETTE[0], linewidth=2, markersize=8,
       label='Average (A2-like)')
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(b) Ordering Energy by Composition', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 3c: Can 2-SL capture the variance?
ax = axes[1, 0]
# Variance at each composition (all configs)
comp_var = df[(df['n_v']>0)&(df['n_v']<8)].groupby('x_V')['dH_f_Jmol_corrected'].var().reset_index()
comp_var.columns = ['x_V', 'var_dHf']
comp_var['std_dHf'] = np.sqrt(comp_var['var_dHf'])

# For 2-SL model, variance comes from different (y_V^A, y_V^B) combinations
# At x_V=0.5 (n_V=4): possible (y_V^A, y_V^B) = (0,1),(0.25,0.75),(0.5,0.5),(0.75,0.25),(1,0)
from itertools import product

def calc_2sl_variance(n_v, coeffs_2sl):
    """Calculate predicted variance from 2-SL model."""
    # All possible (y_V^A, y_V^B) for n_v V atoms on 8 sites (4A + 4B)
    vals = []
    for nv_A in range(min(n_v, 4)+1):
        nv_B = n_v - nv_A
        if nv_B > 4 or nv_B < 0:
            continue
        yA = nv_A / 4.0
        yB = nv_B / 4.0
        # Number of such configs: C(4, nv_A) * C(4, nv_B)
        from scipy.special import comb
        n_configs = comb(4, nv_A, exact=True) * comb(4, nv_B, exact=True)
        # 2-SL prediction
        pred = (coeffs_2sl[0] * (1-yA)*(1-yB) + coeffs_2sl[1] * yA*yB +
                coeffs_2sl[2] * (1-yA)*yB + coeffs_2sl[3] * yA*(1-yB))
        vals.extend([pred] * n_configs)
    return np.var(vals) if vals else 0

# Fit 2-SL model
from numpy.linalg import lstsq as np_lstsq
n_all = len(df)
X_2sl = np.zeros((n_all, 4))
for i, (_, row) in enumerate(df.iterrows()):
    binary = format(int(row['config_index']), '08b')
    atoms = [int(b) for b in binary]
    yA = sum(atoms[0:4]) / 4.0
    yB = sum(atoms[4:8]) / 4.0
    X_2sl[i, 0] = (1-yA) * (1-yB)
    X_2sl[i, 1] = yA * yB
    X_2sl[i, 2] = (1-yA) * yB
    X_2sl[i, 3] = yA * (1-yB)

y_all = df['dH_f_Jmol_corrected'].values
c2sl, _, _, _ = np_lstsq(X_2sl, y_all, rcond=None)

# DFT variance vs 2-SL variance
x_comp = []
var_dft = []
var_2sl = []
for nv in range(1, 8):
    sub = df[df['n_v'] == nv]
    xv = nv / 8.0
    x_comp.append(xv)
    var_dft.append(sub['dH_f_Jmol_corrected'].std()/1000)
    var_2sl.append(np.sqrt(calc_2sl_variance(nv, c2sl))/1000)

width = 0.03
ax.bar(np.array(x_comp) - width/2, var_dft, width=width, color=PALETTE[0], 
      alpha=0.7, label='DFT (actual)', edgecolor='gray')
ax.bar(np.array(x_comp) + width/2, var_2sl, width=width, color=PALETTE[1],
      alpha=0.7, label='2-SL B2 (predicted)', edgecolor='gray')
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'Std Dev of $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(c) Energy Spread: DFT vs 2-SL Model', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 3d: Model accuracy summary
ax = axes[1, 1]
# RK fit to individual configs
def rk_fit_individual(x_arr, dHf_arr, n_terms):
    mask = (x_arr > 0) & (x_arr < 1)
    X = rk_basis(x_arr[mask], n_terms)
    L, _, _, _ = lstsq(X, dHf_arr[mask], rcond=None)
    pred = X @ L
    rmse = np.sqrt(np.mean((dHf_arr[mask] - pred)**2))
    return rmse

# RK fit to composition averages
def rk_fit_avg(x_avg, dHf_avg, n_terms):
    X = rk_basis(x_avg, n_terms)
    L, _, _, _ = lstsq(X, dHf_avg, rcond=None)
    pred = X @ L
    rmse = np.sqrt(np.mean((dHf_avg - pred)**2))
    return rmse

models_names = ['RK-2\n(individual)', 'RK-4\n(individual)', 'RK-2\n(comp. avg)', 'RK-4\n(comp. avg)', 
                '2-SL B2', '2-SL+L B2']
x_all = df['x_V'].values
dHf_all = df['dH_f_Jmol_corrected'].values

# 2-SL extended
X_2sl_ext = np.zeros((n_all, 6))
for i, (_, row) in enumerate(df.iterrows()):
    binary = format(int(row['config_index']), '08b')
    atoms = [int(b) for b in binary]
    yA = sum(atoms[0:4]) / 4.0
    yB = sum(atoms[4:8]) / 4.0
    X_2sl_ext[i, 0] = (1-yA) * (1-yB)
    X_2sl_ext[i, 1] = yA * yB
    X_2sl_ext[i, 2] = (1-yA) * yB
    X_2sl_ext[i, 3] = yA * (1-yB)
    X_2sl_ext[i, 4] = (yA - yB)**2
    X_2sl_ext[i, 5] = (yA - yB)**2 * (yA + yB - 1)

c2sl_ext, _, _, _ = np_lstsq(X_2sl_ext, y_all, rcond=None)
pred_2sl = X_2sl @ c2sl
pred_2sl_ext = X_2sl_ext @ c2sl_ext

rmse_values = [
    rk_fit_individual(x_all, dHf_all, 2)/1000,
    rk_fit_individual(x_all, dHf_all, 4)/1000,
    rk_fit_avg(x_avg, dHf_avg, 2)/1000,
    rk_fit_avg(x_avg, dHf_avg, 4)/1000,
    np.sqrt(np.mean((y_all - pred_2sl)**2))/1000,
    np.sqrt(np.mean((y_all - pred_2sl_ext)**2))/1000,
]

colors_bar = [PALETTE[0]]*2 + [PALETTE[1]]*2 + [PALETTE[2]]*2
bars = ax.barh(models_names, rmse_values, color=colors_bar, alpha=0.8, edgecolor='gray')
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2., 
           f'{val:.1f}', ha='left', va='center', fontsize=ANNOT_SIZE)
ax.set_xlabel('RMSE (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(d) Model RMSE Comparison', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(f'{OUT}/05_interaction_parameters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Regenerated 05_interaction_parameters.png")

# ============================================================
# 4. Print summary for report
# ============================================================
print("\n" + "="*70)
print("Summary for Report")
print("="*70)

# RK fit to composition averages
for n in [2, 4]:
    X = rk_basis(x_avg, n)
    L, _, _, _ = lstsq(X, dHf_avg, rcond=None)
    pred = X @ L
    rmse = np.sqrt(np.mean((dHf_avg - pred)**2))
    r2 = 1 - np.sum((dHf_avg - pred)**2) / np.sum((dHf_avg - np.mean(dHf_avg))**2)
    terms = ', '.join([f'L{v}={L[v]:.0f}' for v in range(n)])
    print(f"RK-{n} (comp avg): RMSE={rmse:.0f}, R²={r2:.4f} | {terms}")

# Ordering energy at each composition
print("\nOrdering energies (E_min - E_avg):")
ord_data = pd.read_csv(f'{OUT}/ordering_energies.csv')
for _, row in ord_data.iterrows():
    print(f"  {row['composition']}: {row['E_ordering']:.0f} J/mol, "
          f"eta_best={row['most_stable_eta']:.2f}")

# Key correlation
print(f"\nGF correlation old vs new: {np.corrcoef(old_vals, new_vals)[0,1]:.4f}")

# Degeneracy: unique energy levels at each composition
print("\nDegeneracy summary:")
for nv in range(1, 8):
    sub = df[df['n_v'] == nv]
    comp = sub['composition'].iloc[0]
    n_unique = len(sub['dH_f_Jmol_corrected'].round(-2).unique())  # round to 100 J
    print(f"  {comp}: {len(sub)} configs, ~{n_unique} unique energy levels")

print("\nDone!")
