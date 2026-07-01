#!/usr/bin/env python3
"""
Generate Fig.A7: FCC 6-pair composition-volume Vegard plot (L1_2 + FCC-SQS).
Analogous to Fig.A6 (BCC) but for FCC structure.
"""

import sys
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from hea_lattice_xgboost import KING_ATOMIC_VOLUMES

# Font setup
for fp in fm.findSystemFonts():
    if "ipag" in fp.lower() or "ipagothic" in fp.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 150,
})

EXCLUDE_ELEMENTS = {
    "Gd", "Ce", "La", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y",
}

OUTDIR = Path(__file__).resolve().parent

# ---- Load L12 data ----
l12_dfs = []
for src in ['MP', 'OQMD', 'VASP']:
    df = pd.read_csv(REPO / f'data/compounds_{src}_L12.csv')
    df['source_db'] = src
    l12_dfs.append(df)
l12_all = pd.concat(l12_dfs, ignore_index=True)
# Compute volume per atom: L12 has 4 atoms per unit cell
l12_all['vol_per_atom'] = l12_all['lattice_constant'] ** 3 / 4.0

# ---- Load FCC SQS data ----
sqs = pd.read_csv(REPO / 'data/sqs_results.csv')
fcc_sqs = sqs[(sqs['structure_root'] == 'FCC_SQS') & (sqs['status'] == 'OK')].copy()
fcc_sqs['vol_per_atom'] = fcc_sqs['volume_A3'] / fcc_sqs['natoms']

# ---- Pure element FCC volumes ----
# From SQS A16A16 with King/MP fallback for >3% deviation
pure_fcc_vol = {}
for _, row in fcc_sqs.iterrows():
    m = re.match(r'([A-Z][a-z]?)16([A-Z][a-z]?)16', row['dir'])
    if m and m.group(1) == m.group(2):
        elem = m.group(1)
        if elem not in EXCLUDE_ELEMENTS:
            pure_fcc_vol[elem] = row['vol_per_atom']

# Also get from L12 homonuclear (A3A = pure element FCC)
for _, row in l12_all.iterrows():
    if row['element_A'] == row['element_B']:
        elem = row['element_A']
        if elem not in EXCLUDE_ELEMENTS and elem not in pure_fcc_vol:
            pure_fcc_vol[elem] = row['vol_per_atom']

# King volumes as reference
king_vol = KING_ATOMIC_VOLUMES

# Override unreliable SQS pure volumes
for elem in list(pure_fcc_vol.keys()):
    sqs_v = pure_fcc_vol[elem]
    ref_v = king_vol.get(elem)
    if ref_v and abs(sqs_v - ref_v) / ref_v > 0.03:
        pure_fcc_vol[elem] = ref_v

# ---- Representative 6 FCC pairs ----
PAIRS = [
    ('Mo', 'Ni'),   # large size mismatch, strong negative Omega
    ('Ni', 'Pt'),   # same group, different period
    ('Ni', 'Ti'),   # 3d-4d, large negative Omega
    ('Pd', 'Pt'),   # noble metal, small mismatch
    ('Pd', 'Zr'),   # large size mismatch
    ('Pt', 'Ti'),   # 5d-3d
]


def get_l12_data(a, b):
    """Get L12 volumes for pair. Returns list of (x_a, vol_per_atom)."""
    results = []
    for _, row in l12_all.iterrows():
        ea, eb = row['element_A'], row['element_B']
        if ea in EXCLUDE_ELEMENTS or eb in EXCLUDE_ELEMENTS:
            continue
        if not ({ea, eb} == {a, b}):
            continue
        if ea == eb:
            continue  # skip homonuclear
        
        vol = row['vol_per_atom']
        count_a = row.get('count_A', 3)
        count_b = row.get('count_B', 1)
        total = count_a + count_b
        
        # x_a is fraction of element 'a' (first in our pair)
        if ea == a:
            x_a = count_a / total
        else:
            x_a = count_b / total
        
        results.append((x_a, vol))
    return results


def get_sqs_data(a, b):
    """Get FCC-SQS data for pair. Returns list of (x_a, vol_per_atom)."""
    results = []
    for _, row in fcc_sqs.iterrows():
        m = re.match(r'([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)', row['dir'])
        if not m:
            continue
        ea, na, eb, nb = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if ea in EXCLUDE_ELEMENTS or eb in EXCLUDE_ELEMENTS:
            continue
        if not ({ea, eb} == {a, b}):
            continue
        if ea == eb:
            continue
        total = na + nb
        if ea == a:
            x_a = na / total
        else:
            x_a = nb / total
        results.append((x_a, row['vol_per_atom']))
    return results


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (a, b) in enumerate(PAIRS):
    ax = axes[idx]
    
    # Pure element volumes for Vegard line
    va = pure_fcc_vol.get(a, king_vol.get(a, 12.0))
    vb = pure_fcc_vol.get(b, king_vol.get(b, 12.0))
    
    # Vegard line: x_a goes from 0 to 1
    x_veg = np.linspace(0, 1, 100)
    v_veg = (1 - x_veg) * vb + x_veg * va
    ax.plot(x_veg, v_veg, 'k--', alpha=0.5, linewidth=1.5, label='Vegard')
    
    # Pure element endpoints
    ax.plot(0, vb, 's', color='gray', markersize=8, zorder=5)
    ax.plot(1, va, 's', color='gray', markersize=8, zorder=5)
    
    # L12 data (red triangles)
    l12_data = get_l12_data(a, b)
    if l12_data:
        l12_x = [d[0] for d in l12_data]
        l12_v = [d[1] for d in l12_data]
        ax.plot(l12_x, l12_v, '^', color='red', markersize=10, alpha=0.7,
                label=f'L1$_2$ (N={len(l12_data)})', zorder=4)
    
    # FCC-SQS data (green diamonds)
    sqs_data = get_sqs_data(a, b)
    if sqs_data:
        sqs_x = [d[0] for d in sqs_data]
        sqs_v = [d[1] for d in sqs_data]
        ax.plot(sqs_x, sqs_v, 'D', color='green', markersize=10, alpha=0.7,
                label=f'FCC-SQS (N={len(sqs_data)})', zorder=4)
    
    # Omega annotations
    if sqs_data:
        sqs_50 = [d for d in sqs_data if abs(d[0] - 0.5) < 0.05]
        if sqs_50:
            v_sqs_50 = np.mean([d[1] for d in sqs_50])
            v_veg_50 = 0.5 * va + 0.5 * vb
            omega_sqs = (v_sqs_50 - v_veg_50) / v_veg_50
            ax.text(0.05, 0.05, f'$\\Omega_{{\\mathrm{{SQS}}}}$ = {omega_sqs:+.3f}',
                    transform=ax.transAxes, fontsize=13, color='green',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if l12_data:
        # Use 25% composition L12 data for Omega
        l12_25 = [d for d in l12_data if abs(d[0] - 0.25) < 0.05]
        if l12_25:
            v_l12 = np.mean([d[1] for d in l12_25])
            v_veg_25 = 0.75 * vb + 0.25 * va
            omega_l12 = (v_l12 - v_veg_25) / v_veg_25
        else:
            l12_75 = [d for d in l12_data if abs(d[0] - 0.75) < 0.05]
            if l12_75:
                v_l12 = np.mean([d[1] for d in l12_75])
                v_veg_75 = 0.25 * vb + 0.75 * va
                omega_l12 = (v_l12 - v_veg_75) / v_veg_75
            else:
                omega_l12 = None
        
        if omega_l12 is not None:
            ax.text(0.05, 0.15, f'$\\Omega_{{\\mathrm{{L1}}_2}}$ = {omega_l12:+.3f}',
                    transform=ax.transAxes, fontsize=13, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{a}–{b}', fontsize=18)
    ax.set_xlabel(f'$x_{{{a}}}$', fontsize=16)
    ax.set_ylabel('V [Å³/atom]', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
outpath = OUTDIR / 'fig_vegard_pairs_fcc_l12_sqs.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath}")

# Print summary
for a, b in PAIRS:
    l12_data = get_l12_data(a, b)
    sqs_data = get_sqs_data(a, b)
    print(f"  {a}-{b}: L12={len(l12_data)}, SQS={len(sqs_data)}")
