#!/usr/bin/env python3
"""
Fe-V B2_221 Detailed Analysis Report
- SRO (Short-Range Order) analysis
- TDB interaction parameter analysis
- Metastable B2 reproducibility via interaction coefficient fitting
- Comparison of old vs new DFT datasets
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares, curve_fit
from scipy.special import comb
from itertools import combinations
import os, warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")

TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 12
ANNOT_SIZE = 11

OUT = "/tmp/fev_detailed_report"
os.makedirs(OUT, exist_ok=True)

DATA_PATH = "/home/ubuntu/repos/machine-learning/calphad/fev_vasp_corrected_data.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} configurations")
print(f"Columns: {list(df.columns)}")

# ============================================================
# 1. Warren-Cowley SRO Parameter Calculation
# ============================================================
print("\n" + "="*70)
print("1. SRO (Short-Range Order) Analysis")
print("="*70)

def calc_warren_cowley_sro(config_idx, n_atoms=8):
    """
    Calculate Warren-Cowley SRO parameter alpha_1 for B2 2x2x1 supercell.
    
    B2 structure: corner sites (sublattice A) and body-center sites (sublattice B)
    In 2x2x1 supercell with 8 atoms:
    - Sites 0-3 (bits 7,6,5,4 in binary) = sublattice A (corners)
    - Sites 4-7 (bits 3,2,1,0 in binary) = sublattice B (body-centers)
    
    Each A site has 8 nearest neighbors (all B sites in BCC)
    In 2x2x1 supercell with PBC, each A site has exactly 4 B-site neighbors.
    
    Warren-Cowley: alpha_1 = 1 - P_AB / x_B
    where P_AB = probability of finding B-type atom on NN shell of A-type atom
    """
    binary = format(config_idx, '08b')
    # sites: binary[0]=SL1(A), binary[1]=SL2(A), binary[2]=SL3(A), binary[3]=SL4(A)
    #        binary[4]=SL5(B), binary[5]=SL6(B), binary[6]=SL7(B), binary[7]=SL8(B)
    
    # V=1, Fe=0 in binary representation
    atoms = [int(b) for b in binary]  # 1=V, 0=Fe
    
    sl_A = atoms[0:4]  # corner sublattice
    sl_B = atoms[4:8]  # body-center sublattice
    
    x_V = sum(atoms) / 8.0
    if x_V == 0 or x_V == 1:
        return 0.0  # No mixing, SRO undefined
    
    # In B2 BCC, NN pairs are A-B pairs
    # In 2x2x1 supercell, each A site connects to multiple B sites
    # For the BCC lattice with 2x2x1 supercell:
    # NN connectivity (with PBC): each A site has 8 NN, all on B sublattice
    # Due to 2x2x1 periodicity, some B sites appear multiple times
    # 
    # Simplified: count unlike A-B pairs
    n_AB_unlike = 0
    n_AB_total = 0
    for a in sl_A:
        for b in sl_B:
            n_AB_total += 1
            if a != b:  # unlike pair (Fe-V or V-Fe)
                n_AB_unlike += 1
    
    # P_AB(V) = fraction of V on B-sublattice NN of an A-site atom
    # For random alloy: P_AB(V) = x_V
    # SRO = 1 - P_unlike / (2 * x_V * (1-x_V))
    p_unlike = n_AB_unlike / n_AB_total
    p_random = 2 * x_V * (1 - x_V)
    
    if p_random == 0:
        return 0.0
    
    alpha = 1 - p_unlike / p_random
    return alpha

def calc_sublattice_segregation(config_idx):
    """
    Calculate sublattice segregation order parameter.
    eta = |y_V^A - y_V^B| where y_V^A, y_V^B are V fractions on sublattices A, B.
    For perfect B2 order: eta = 1
    For random: eta ~ 0
    """
    binary = format(config_idx, '08b')
    atoms = [int(b) for b in binary]
    sl_A = atoms[0:4]
    sl_B = atoms[4:8]
    y_V_A = sum(sl_A) / 4.0
    y_V_B = sum(sl_B) / 4.0
    eta = abs(y_V_A - y_V_B)
    return eta, y_V_A, y_V_B

# Calculate SRO for all configs
df['sro_alpha'] = df['config_index'].apply(calc_warren_cowley_sro)
sro_data = df['config_index'].apply(lambda idx: calc_sublattice_segregation(idx))
df['eta'] = [s[0] for s in sro_data]
df['y_V_A'] = [s[1] for s in sro_data]
df['y_V_B'] = [s[2] for s in sro_data]

# SRO statistics by composition
print("\nSRO statistics by composition:")
for comp in sorted(df['composition'].unique()):
    sub = df[df['composition'] == comp]
    n_v = sub['n_v'].iloc[0]
    if n_v == 0 or n_v == 8:
        continue
    print(f"  {comp}: alpha = {sub['sro_alpha'].mean():.4f} +/- {sub['sro_alpha'].std():.4f}, "
          f"eta = {sub['eta'].mean():.4f} +/- {sub['eta'].std():.4f}")

# ============================================================
# 2. Effective Pair Interaction (EPI) Analysis
# ============================================================
print("\n" + "="*70)
print("2. Effective Pair Interaction (EPI) Analysis")
print("="*70)

# In the Ising model on BCC lattice:
# E_config = E_0 + sum_i(V_i * sigma_i) + sum_{i<j}(J_ij * sigma_i * sigma_j)
# where sigma_i = +1 (V) or -1 (Fe)
# 
# For 8-site supercell with 2 sublattices (A: sites 0-3, B: sites 4-7):
# We can extract effective cluster interactions from DFT formation energies

def get_spin_config(config_idx):
    """Convert config index to Ising spins: Fe=-1, V=+1"""
    binary = format(config_idx, '08b')
    return np.array([1 if b == '1' else -1 for b in binary])

# Build correlation function matrix for cluster expansion
# Clusters: point (8), NN pairs (A-B: 16), NNN pairs (A-A: 6, B-B: 6)
def build_cluster_matrix(df):
    """Build cluster correlation matrix for all configurations."""
    n = len(df)
    
    # Point correlations: <sigma_i> for each site
    # NN pair correlations: <sigma_i * sigma_j> for NN pairs (A-B)
    # NNN pair correlations: <sigma_i * sigma_j> for NNN pairs (A-A, B-B)
    
    # For simplicity, use symmetry-reduced clusters:
    # 1. Empty cluster (constant = 1)
    # 2. Point cluster (average spin = composition)
    # 3. NN pair (A-B average)
    # 4. NNN pair A-A (average)
    # 5. NNN pair B-B (average)
    # 6. Triangle clusters...
    
    # Simpler approach: fit Redlich-Kister polynomial to composition-averaged energies
    # and then analyze residuals as function of SRO
    
    correlations = np.zeros((n, 6))
    
    for i, row in df.iterrows():
        sigma = get_spin_config(row['config_index'])
        sl_A = sigma[0:4]
        sl_B = sigma[4:8]
        
        # Cluster 0: constant
        correlations[i, 0] = 1.0
        
        # Cluster 1: point (overall magnetization in Ising sense)
        correlations[i, 1] = np.mean(sigma)
        
        # Cluster 2: NN pairs (A-B)
        nn_corr = 0
        count = 0
        for a in sl_A:
            for b in sl_B:
                nn_corr += a * b
                count += 1
        correlations[i, 2] = nn_corr / count
        
        # Cluster 3: NNN pairs A-A
        nnn_A = 0
        count_A = 0
        for j in range(4):
            for k in range(j+1, 4):
                nnn_A += sl_A[j] * sl_A[k]
                count_A += 1
        correlations[i, 3] = nnn_A / count_A if count_A > 0 else 0
        
        # Cluster 4: NNN pairs B-B
        nnn_B = 0
        count_B = 0
        for j in range(4):
            for k in range(j+1, 4):
                nnn_B += sl_B[j] * sl_B[k]
                count_B += 1
        correlations[i, 4] = nnn_B / count_B if count_B > 0 else 0
        
        # Cluster 5: triplet (A-A-B average)
        trip = 0
        count_t = 0
        for j in range(4):
            for k in range(j+1, 4):
                for l in range(4):
                    trip += sl_A[j] * sl_A[k] * sl_B[l]
                    count_t += 1
        correlations[i, 5] = trip / count_t if count_t > 0 else 0
    
    return correlations

corr_matrix = build_cluster_matrix(df)

# Fit cluster expansion to DFT formation energies
from numpy.linalg import lstsq

# Use corrected formation energies
y = df['dH_f_Jmol_corrected'].values

# Fit with different numbers of clusters
for n_clusters in [3, 4, 5, 6]:
    X = corr_matrix[:, :n_clusters]
    coeffs, residuals, rank, sv = lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    print(f"  {n_clusters} clusters: RMSE = {rmse:.1f} J/mol, R² = {r2:.4f}")
    if n_clusters == 3:
        labels = ['const', 'point', 'NN_pair']
    elif n_clusters == 4:
        labels = ['const', 'point', 'NN_pair', 'NNN_AA']
    elif n_clusters == 5:
        labels = ['const', 'point', 'NN_pair', 'NNN_AA', 'NNN_BB']
    elif n_clusters == 6:
        labels = ['const', 'point', 'NN_pair', 'NNN_AA', 'NNN_BB', 'triplet']
    for l, c in zip(labels, coeffs):
        print(f"    {l:12s} = {c:.1f} J/mol")

# Full 6-cluster fit for further analysis
X_full = corr_matrix[:, :6]
coeffs_full, _, _, _ = lstsq(X_full, y, rcond=None)
y_pred_full = X_full @ coeffs_full
residuals_full = y - y_pred_full

# ============================================================
# 3. Redlich-Kister Analysis (BCC_A2 comparison)
# ============================================================
print("\n" + "="*70)
print("3. Redlich-Kister Analysis for BCC_A2")
print("="*70)

# BCC_A2 uses Redlich-Kister: G_mix = x_Fe*x_V * sum_v(L_v * (x_Fe-x_V)^v)
# From TDB: L0 = -21427 + 6.846*T, L1 = 7345 - 1.509*T
# At 0K (DFT): L0_0K = -21427, L1_0K = 7345

# DFT formation energy in regular solution model:
# dH_f = x_Fe * x_V * (L0 + L1*(x_Fe - x_V) + L2*(x_Fe - x_V)^2 + ...)

x_V = df['x_V'].values
x_Fe = 1 - x_V

# Composition-averaged formation energies (converged only)
df_conv = df[df['converged'] == True].copy()
comp_avg = df_conv.groupby('x_V')['dH_f_Jmol_corrected'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
comp_avg.columns = ['x_V', 'mean_dHf', 'std_dHf', 'min_dHf', 'max_dHf', 'count']

print("\nComposition-averaged formation energies (converged):")
for _, row in comp_avg.iterrows():
    print(f"  x_V = {row['x_V']:.3f}: mean = {row['mean_dHf']:.0f} J/mol, "
          f"min = {row['min_dHf']:.0f}, max = {row['max_dHf']:.0f} (n={int(row['count'])})")

# Fit Redlich-Kister to composition-averaged data
# G_mix_avg(x_V) = x_Fe * x_V * sum_v(L_v * (x_Fe - x_V)^v)
# We need to find L_v coefficients

# Use all DFT data (not just averages) for better statistics
mask_mix = (x_V > 0) & (x_V < 1)
x_V_mix = x_V[mask_mix]
x_Fe_mix = x_Fe[mask_mix]
dHf_mix = df['dH_f_Jmol_corrected'].values[mask_mix]

def rk_basis(x_V_arr, n_terms):
    """Build Redlich-Kister basis: x_Fe*x_V*(x_Fe-x_V)^v for v=0,1,...,n_terms-1"""
    x_Fe_arr = 1 - x_V_arr
    X = np.zeros((len(x_V_arr), n_terms))
    for v in range(n_terms):
        X[:, v] = x_Fe_arr * x_V_arr * (x_Fe_arr - x_V_arr)**v
    return X

# Fit RK with 1-4 terms using ALL individual config data
print("\nRedlich-Kister fits (all individual configs, composition-average dH_f):")
rk_results = {}
for n_terms in [1, 2, 3, 4]:
    X_rk = rk_basis(x_V_mix, n_terms)
    L_coeffs, res, _, _ = lstsq(X_rk, dHf_mix, rcond=None)
    dHf_pred = X_rk @ L_coeffs
    rmse = np.sqrt(np.mean((dHf_mix - dHf_pred)**2))
    r2 = 1 - np.sum((dHf_mix - dHf_pred)**2) / np.sum((dHf_mix - np.mean(dHf_mix))**2)
    rk_results[n_terms] = {'L': L_coeffs, 'rmse': rmse, 'r2': r2, 'pred': dHf_pred}
    terms_str = ', '.join([f'L{v}={L_coeffs[v]:.0f}' for v in range(n_terms)])
    print(f"  {n_terms} terms: RMSE={rmse:.0f} J/mol, R²={r2:.4f} | {terms_str}")

# Compare with Hari Kumar & Raghavan parameters
L0_HKR = -21427  # J/mol (at 0K)
L1_HKR = 7345   # J/mol (at 0K)
print(f"\nHari Kumar & Raghavan (1991) at 0K: L0 = {L0_HKR}, L1 = {L1_HKR}")
print(f"DFT fit (2 terms):                  L0 = {rk_results[2]['L'][0]:.0f}, L1 = {rk_results[2]['L'][1]:.0f}")

# ============================================================
# 4. SRO-Energy Correlation Analysis
# ============================================================
print("\n" + "="*70)
print("4. SRO-Energy Correlation")
print("="*70)

# For each composition, calculate energy deviation from RK prediction
# This deviation captures the ordering energy
X_rk_all = rk_basis(x_V, 2)
# Need to handle x_V=0 and x_V=1 (where RK = 0)
dHf_rk_pred_all = np.zeros(len(x_V))
mask_mix_all = (x_V > 0) & (x_V < 1)
dHf_rk_pred_all[mask_mix_all] = rk_basis(x_V[mask_mix_all], 2) @ rk_results[2]['L']
df['dHf_rk_pred'] = dHf_rk_pred_all
df['dHf_deviation'] = df['dH_f_Jmol_corrected'] - df['dHf_rk_pred']

# Correlation between SRO and energy deviation
for comp in sorted(df['composition'].unique()):
    sub = df[(df['composition'] == comp) & (df['n_v'] > 0) & (df['n_v'] < 8)]
    if len(sub) < 3:
        continue
    corr_sro = sub['sro_alpha'].corr(sub['dHf_deviation'])
    corr_eta = sub['eta'].corr(sub['dH_f_Jmol_corrected'])
    print(f"  {comp}: corr(SRO, dE_deviation)={corr_sro:.3f}, corr(eta, dHf)={corr_eta:.3f}")

# ============================================================
# 5. B2 Order-Disorder Analysis
# ============================================================
print("\n" + "="*70)
print("5. B2 Order-Disorder Transition Analysis")
print("="*70)

# For equiatomic Fe4V4 (x_V = 0.5):
# Perfectly ordered B2: Fe on A, V on B (or vice versa) -> eta = 1
# Random A2: eta = 0
# 
# In 2x2x1 supercell: 
# Perfect B2 = config 00001111 = 15 or 11110000 = 240

df_eq = df[df['n_v'] == 4].copy()
print(f"\nEquiatomic Fe4V4 configurations: {len(df_eq)}")
print(f"  eta range: {df_eq['eta'].min():.2f} to {df_eq['eta'].max():.2f}")
print(f"  Formation energy range: {df_eq['dH_f_Jmol_corrected'].min():.0f} to {df_eq['dH_f_Jmol_corrected'].max():.0f} J/mol")

# Group by eta
for eta_val in sorted(df_eq['eta'].unique()):
    sub = df_eq[df_eq['eta'] == eta_val]
    print(f"  eta={eta_val:.2f}: n={len(sub)}, mean_dHf={sub['dH_f_Jmol_corrected'].mean():.0f} J/mol, "
          f"std={sub['dH_f_Jmol_corrected'].std():.0f}")

# Identify the B2-ordered configurations
# Perfect B2: all A sites = Fe (0000), all B sites = V (1111) -> 00001111 = 15
# Or: all A sites = V (1111), all B sites = Fe (0000) -> 11110000 = 240
b2_perfect = df[(df['config_index'] == 15) | (df['config_index'] == 240)]
print("\nPerfect B2 configurations:")
for _, row in b2_perfect.iterrows():
    binary = format(row['config_index'], '08b')
    print(f"  config_{row['config_index']:03d} ({binary}): dHf = {row['dH_f_Jmol_corrected']:.0f} J/mol, eta = {row['eta']:.2f}")

# ============================================================
# 6. Ordering Energy Estimation
# ============================================================
print("\n" + "="*70)
print("6. Ordering Energy Analysis")
print("="*70)

# Ordering energy = E_ordered - E_random (at same composition)
# E_random approximated by composition-average
# E_ordered = minimum energy configuration at that composition

for comp in sorted(df['composition'].unique()):
    sub = df[df['composition'] == comp]
    n_v = sub['n_v'].iloc[0]
    if n_v == 0 or n_v == 8:
        continue
    
    E_avg = sub['dH_f_Jmol_corrected'].mean()
    E_min = sub['dH_f_Jmol_corrected'].min()
    E_max = sub['dH_f_Jmol_corrected'].max()
    E_ordered = E_min  # most stable = most ordered
    E_ordering = E_min - E_avg
    
    # Find most ordered config
    most_stable_idx = sub['dH_f_Jmol_corrected'].idxmin()
    ms = sub.loc[most_stable_idx]
    
    print(f"  {comp}: E_avg={E_avg:.0f}, E_min={E_min:.0f}, "
          f"E_ordering={E_ordering:.0f} J/mol (config_{int(ms['config_index']):03d}, eta={ms['eta']:.2f})")

# ============================================================
# 7. Can interaction parameters reproduce metastable B2?
# ============================================================
print("\n" + "="*70)
print("7. Metastable B2 Reproducibility Analysis")
print("="*70)

# Key question: Can we reproduce the B2 phase stability using
# BCC_A2 interaction parameters + ordering contribution?
#
# In CALPHAD, the A2/B2 transition can be modeled as:
# G_B2 = G_A2 + G_ord(eta, T)
# where G_ord is the ordering contribution
#
# The 8-sublattice CEF approach explicitly represents all configurations
# Alternative: 2-sublattice B2 model with interaction parameters
#
# Compare: 
# (a) Full 256-endmember CEF (current approach)
# (b) 2-sublattice B2 with fitted W parameter
# (c) RK polynomial (no ordering)

# 2-sublattice B2 model:
# G_B2 = y_V^A * y_V^B * G_VV + y_Fe^A * y_Fe^B * G_FeFe 
#       + y_Fe^A * y_V^B * G_FeV + y_V^A * y_Fe^B * G_VFe
#       + RT * sum(y_i * ln(y_i))
# 
# At 0K, no entropy: G_B2 = sum(y_i^A * y_j^B * G_ij)
# The ordering energy comes from G_FeV != G_VFe (asymmetric)

# Fit 2-sublattice model to DFT data
# G(y_V^A, y_V^B) = (1-y_V^A)*(1-y_V^B)*G_FeFe + y_V^A*y_V^B*G_VV 
#                   + (1-y_V^A)*y_V^B*G_FeV + y_V^A*(1-y_V^B)*G_VFe
#                   + L_AB * (y_V^A - y_V^B)^2  [optional interaction]

# For 8-sublattice: y_V^A = mean(V on sites 0-3), y_V^B = mean(V on sites 4-7)
# But this is a coarse-graining: many 8-SL configs map to same (y_V^A, y_V^B)

def build_2sl_matrix(df):
    """Build design matrix for 2-sublattice B2 model."""
    n = len(df)
    # Basis: G_FeFe, G_VV, G_FeV, G_VFe
    X = np.zeros((n, 4))
    for i, (_, row) in enumerate(df.iterrows()):
        yA = row['y_V_A']
        yB = row['y_V_B']
        X[i, 0] = (1-yA) * (1-yB)  # G_FeFe
        X[i, 1] = yA * yB          # G_VV
        X[i, 2] = (1-yA) * yB      # G_FeV
        X[i, 3] = yA * (1-yB)      # G_VFe
    return X

X_2sl = build_2sl_matrix(df)
y_dHf = df['dH_f_Jmol_corrected'].values

coeffs_2sl, res_2sl, _, _ = lstsq(X_2sl, y_dHf, rcond=None)
y_pred_2sl = X_2sl @ coeffs_2sl
rmse_2sl = np.sqrt(np.mean((y_dHf - y_pred_2sl)**2))
r2_2sl = 1 - np.sum((y_dHf - y_pred_2sl)**2) / np.sum((y_dHf - np.mean(y_dHf))**2)

print("\n2-sublattice B2 model:")
print(f"  G_FeFe = {coeffs_2sl[0]:.0f} J/mol")
print(f"  G_VV   = {coeffs_2sl[1]:.0f} J/mol")
print(f"  G_FeV  = {coeffs_2sl[2]:.0f} J/mol")
print(f"  G_VFe  = {coeffs_2sl[3]:.0f} J/mol")
print(f"  RMSE   = {rmse_2sl:.0f} J/mol")
print(f"  R²     = {r2_2sl:.4f}")
print(f"  Ordering energy at x_V=0.5: {(coeffs_2sl[2]+coeffs_2sl[3])/2 - (coeffs_2sl[0]+coeffs_2sl[1])/2:.0f} J/mol")

# Extended 2-sublattice model with L parameters
def build_2sl_extended(df):
    """Build design matrix for 2-sublattice B2 model with interaction parameters."""
    n = len(df)
    # Basis: G_FeFe, G_VV, G_FeV, G_VFe, L0*(yA-yB)^2, L1*(yA-yB)^2*(yA+yB-1)
    X = np.zeros((n, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        yA = row['y_V_A']
        yB = row['y_V_B']
        X[i, 0] = (1-yA) * (1-yB)
        X[i, 1] = yA * yB
        X[i, 2] = (1-yA) * yB
        X[i, 3] = yA * (1-yB)
        X[i, 4] = (yA - yB)**2
        X[i, 5] = (yA - yB)**2 * (yA + yB - 1)
    return X

X_2sl_ext = build_2sl_extended(df)
coeffs_2sl_ext, _, _, _ = lstsq(X_2sl_ext, y_dHf, rcond=None)
y_pred_2sl_ext = X_2sl_ext @ coeffs_2sl_ext
rmse_2sl_ext = np.sqrt(np.mean((y_dHf - y_pred_2sl_ext)**2))
r2_2sl_ext = 1 - np.sum((y_dHf - y_pred_2sl_ext)**2) / np.sum((y_dHf - np.mean(y_dHf))**2)

print("\n2-sublattice B2 model (extended with L parameters):")
print(f"  G_FeFe = {coeffs_2sl_ext[0]:.0f} J/mol")
print(f"  G_VV   = {coeffs_2sl_ext[1]:.0f} J/mol")
print(f"  G_FeV  = {coeffs_2sl_ext[2]:.0f} J/mol")
print(f"  G_VFe  = {coeffs_2sl_ext[3]:.0f} J/mol")
print(f"  L0     = {coeffs_2sl_ext[4]:.0f} J/mol")
print(f"  L1     = {coeffs_2sl_ext[5]:.0f} J/mol")
print(f"  RMSE   = {rmse_2sl_ext:.0f} J/mol")
print(f"  R²     = {r2_2sl_ext:.4f}")

# Full 8-sublattice CEF (identity, trivially perfect)
print("\n8-sublattice CEF (256 endmembers): RMSE = 0 J/mol, R² = 1.0000")

# ============================================================
# 8. Degeneracy Analysis
# ============================================================
print("\n" + "="*70)
print("8. Configuration Degeneracy Analysis")
print("="*70)

# How many unique energy levels exist?
# Configs related by symmetry should have same energy
energy_tolerance = 100  # J/mol
df_sorted = df.sort_values('dH_f_Jmol_corrected')

unique_energies = []
for _, row in df_sorted.iterrows():
    e = row['dH_f_Jmol_corrected']
    found = False
    for ue in unique_energies:
        if abs(e - ue['energy']) < energy_tolerance:
            ue['count'] += 1
            ue['configs'].append(row['config_index'])
            found = True
            break
    if not found:
        unique_energies.append({'energy': e, 'count': 1, 'configs': [row['config_index']]})

print(f"Number of unique energy levels (tolerance={energy_tolerance} J/mol): {len(unique_energies)}")
print("\nTop 10 most degenerate levels:")
ue_sorted = sorted(unique_energies, key=lambda x: -x['count'])
for ue in ue_sorted[:10]:
    # Find composition
    idx = ue['configs'][0]
    comp = df[df['config_index'] == idx]['composition'].iloc[0]
    print(f"  E = {ue['energy']:.0f} J/mol, degeneracy = {ue['count']}, comp = {comp}")

# ============================================================
# 9. Comparison: Old vs New TDB datasets
# ============================================================
print("\n" + "="*70)
print("9. Old vs New Dataset Comparison (structural)")
print("="*70)

# Read old TDB GF values
old_tdb_path = "/home/ubuntu/repos/machine-learning/calphad/Fe-V_B2_221.tdb"
old_gf = {}
with open(old_tdb_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('FUNCTION GF') and '298.15' in line:
            parts = line.split()
            name = parts[1]
            val_str = parts[2]
            val = float(val_str.replace(';', ''))
            old_gf[name] = val

new_gf = {}
new_tdb_path = "/home/ubuntu/repos/machine-learning/calphad/Fe-V_B2_221_VASP.tdb"
with open(new_tdb_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('FUNCTION GF') and '298.15' in line:
            parts = line.split()
            name = parts[1]
            val_str = parts[2]
            val = float(val_str.replace(';', ''))
            new_gf[name] = val

# Compare common GF functions
common_funcs = set(old_gf.keys()) & set(new_gf.keys())
print(f"Old TDB: {len(old_gf)} GF functions")
print(f"New TDB: {len(new_gf)} GF functions")
print(f"Common: {len(common_funcs)} functions")

if common_funcs:
    diffs = []
    for func in sorted(common_funcs):
        diff = new_gf[func] - old_gf[func]
        diffs.append(diff)
    diffs = np.array(diffs)
    print("\nDifferences (New - Old) for common functions:")
    print(f"  Mean  = {np.mean(diffs):.0f} J/mol")
    print(f"  Std   = {np.std(diffs):.0f} J/mol")
    print(f"  Min   = {np.min(diffs):.0f} J/mol")
    print(f"  Max   = {np.max(diffs):.0f} J/mol")

# ============================================================
# FIGURES
# ============================================================
print("\n" + "="*70)
print("Generating Figures...")
print("="*70)

# --- Figure 1: SRO Parameter Distribution ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1a: SRO vs composition
ax = axes[0, 0]
scatter = ax.scatter(df['x_V'], df['sro_alpha'], c=df['dH_f_Jmol_corrected']/1000, 
                     cmap='RdYlBu_r', s=40, alpha=0.7, edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'Warren-Cowley SRO ($\alpha_1$)', fontsize=LABEL_SIZE)
ax.set_title('(a) SRO vs Composition', fontsize=TITLE_SIZE)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.tick_params(labelsize=TICK_SIZE)

# 1b: eta vs composition
ax = axes[0, 1]
scatter = ax.scatter(df['x_V'], df['eta'], c=df['dH_f_Jmol_corrected']/1000,
                     cmap='RdYlBu_r', s=40, alpha=0.7, edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'Sublattice Order Parameter ($\eta$)', fontsize=LABEL_SIZE)
ax.set_title(r'(b) $\eta$ vs Composition', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 1c: SRO vs formation energy (colored by composition)
ax = axes[1, 0]
for comp in sorted(df['composition'].unique()):
    sub = df[(df['composition'] == comp) & (df['n_v'] > 0) & (df['n_v'] < 8)]
    if len(sub) == 0:
        continue
    ax.scatter(sub['sro_alpha'], sub['dH_f_Jmol_corrected']/1000, 
              label=comp, s=30, alpha=0.7)
ax.set_xlabel(r'Warren-Cowley SRO ($\alpha_1$)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(c) SRO vs Formation Energy', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE-2, ncol=2, loc='upper right')
ax.tick_params(labelsize=TICK_SIZE)

# 1d: eta vs formation energy for Fe4V4
ax = axes[1, 1]
df_eq_plot = df[df['n_v'] == 4].copy()
ax.scatter(df_eq_plot['eta'], df_eq_plot['dH_f_Jmol_corrected']/1000, 
          c=PALETTE[0], s=60, alpha=0.7, edgecolors='gray', linewidth=0.5)
# Add trend line
if len(df_eq_plot) > 2:
    z = np.polyfit(df_eq_plot['eta'], df_eq_plot['dH_f_Jmol_corrected']/1000, 2)
    x_fit = np.linspace(df_eq_plot['eta'].min(), df_eq_plot['eta'].max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), 'r--', linewidth=2, label='Polynomial fit')
ax.set_xlabel(r'Sublattice Order Parameter ($\eta$)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(r'(d) Fe$_4$V$_4$: $\eta$ vs $\Delta H_f$', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(f'{OUT}/01_sro_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01_sro_analysis.png")

# --- Figure 2: Cluster Expansion Fit Quality ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 2a: 6-cluster fit parity plot
ax = axes[0, 0]
ax.scatter(y_dHf/1000, y_pred_full/1000, c=df['x_V'], cmap='viridis', 
          s=30, alpha=0.6, edgecolors='gray', linewidth=0.3)
lim = [min(y_dHf.min(), y_pred_full.min())/1000 - 5, max(y_dHf.max(), y_pred_full.max())/1000 + 5]
ax.plot(lim, lim, 'r--', linewidth=1.5)
ax.set_xlabel(r'DFT $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'Cluster Expansion $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(f'(a) 6-Cluster CE Fit (R²={1 - np.sum((y_dHf - y_pred_full)**2)/np.sum((y_dHf - np.mean(y_dHf))**2):.4f})', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 2b: 2-sublattice model parity
ax = axes[0, 1]
ax.scatter(y_dHf/1000, y_pred_2sl/1000, c=df['x_V'], cmap='viridis',
          s=30, alpha=0.6, edgecolors='gray', linewidth=0.3)
ax.plot(lim, lim, 'r--', linewidth=1.5)
ax.set_xlabel(r'DFT $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'2-SL Model $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(f'(b) 2-Sublattice B2 (R²={r2_2sl:.4f})', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 2c: Redlich-Kister parity (2-term)
ax = axes[1, 0]
dHf_rk2_pred = np.zeros(len(df))
mask = (df['x_V'] > 0) & (df['x_V'] < 1)
dHf_rk2_pred[mask] = rk_basis(df['x_V'].values[mask], 2) @ rk_results[2]['L']
ax.scatter(y_dHf/1000, dHf_rk2_pred/1000, c=df['x_V'], cmap='viridis',
          s=30, alpha=0.6, edgecolors='gray', linewidth=0.3)
ax.plot(lim, lim, 'r--', linewidth=1.5)
ax.set_xlabel(r'DFT $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'RK (2-term) $\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
rmse_rk2 = np.sqrt(np.mean((y_dHf[mask] - dHf_rk2_pred[mask])**2))
r2_rk2 = rk_results[2]['r2']
ax.set_title(f'(c) Redlich-Kister 2-term (R²={r2_rk2:.4f})', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 2d: Model comparison bar chart
ax = axes[1, 1]
models = ['RK-1', 'RK-2', 'RK-3', 'RK-4', '2-SL B2', '2-SL+L', '6-Cluster\nCE', '8-SL CEF\n(256 params)']
rmse_values = [
    rk_results[1]['rmse']/1000,
    rk_results[2]['rmse']/1000,
    rk_results[3]['rmse']/1000,
    rk_results[4]['rmse']/1000,
    rmse_2sl/1000,
    rmse_2sl_ext/1000,
    np.sqrt(np.mean(residuals_full**2))/1000,
    0
]
r2_values = [
    rk_results[1]['r2'],
    rk_results[2]['r2'],
    rk_results[3]['r2'],
    rk_results[4]['r2'],
    r2_2sl,
    r2_2sl_ext,
    1 - np.sum(residuals_full**2)/np.sum((y_dHf - np.mean(y_dHf))**2),
    1.0
]
n_params = [1, 2, 3, 4, 4, 6, 6, 256]

colors_bar = [PALETTE[0]]*4 + [PALETTE[1]]*2 + [PALETTE[2]] + [PALETTE[3]]
bars = ax.bar(models, rmse_values, color=colors_bar, alpha=0.8, edgecolor='gray')
ax.set_ylabel('RMSE (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(d) Model Comparison', fontsize=TITLE_SIZE)
ax.tick_params(axis='x', labelsize=TICK_SIZE-2, rotation=30)
ax.tick_params(axis='y', labelsize=TICK_SIZE)
# Add R² annotations
for bar, r2, np_ in zip(bars, r2_values, n_params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
           f'R²={r2:.3f}\n(n={np_})', ha='center', va='bottom', fontsize=ANNOT_SIZE-1)

plt.tight_layout()
plt.savefig(f'{OUT}/02_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 02_model_comparison.png")

# --- Figure 3: Ordering Energy Landscape ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 3a: Formation energy colored by eta
ax = axes[0, 0]
scatter = ax.scatter(df['x_V'], df['dH_f_Jmol_corrected']/1000, 
                    c=df['eta'], cmap='plasma', s=40, alpha=0.7,
                    edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$\eta$ (order parameter)', fontsize=LABEL_SIZE)
# Plot composition-averaged RK prediction
x_plot = np.linspace(0.01, 0.99, 200)
dHf_rk_plot = x_plot * (1-x_plot) * (rk_results[2]['L'][0] + rk_results[2]['L'][1]*(1-2*x_plot))
ax.plot(x_plot, dHf_rk_plot/1000, 'k--', linewidth=2, label='RK fit (A2 disordered)')
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(r'(a) Formation Energy colored by $\eta$', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 3b: Energy deviation from RK (= ordering contribution)
ax = axes[0, 1]
df_mix = df[(df['n_v'] > 0) & (df['n_v'] < 8)].copy()
scatter = ax.scatter(df_mix['x_V'], df_mix['dHf_deviation']/1000,
                    c=df_mix['eta'], cmap='plasma', s=40, alpha=0.7,
                    edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$\eta$', fontsize=LABEL_SIZE)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f - \Delta H_f^{RK}$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(b) Energy Deviation from RK (Ordering Energy)', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 3c: Ordering energy distribution by composition
ax = axes[1, 0]
compositions = []
ordering_energies = []
for comp in sorted(df['composition'].unique()):
    sub = df[df['composition'] == comp]
    if sub['n_v'].iloc[0] == 0 or sub['n_v'].iloc[0] == 8:
        continue
    for _, row in sub.iterrows():
        compositions.append(comp)
        ordering_energies.append(row['dHf_deviation']/1000)

ord_df = pd.DataFrame({'Composition': compositions, 'Ordering Energy (kJ/mol)': ordering_energies})
sns.boxplot(data=ord_df, x='Composition', y='Ordering Energy (kJ/mol)', ax=ax, palette='Set2')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Composition', fontsize=LABEL_SIZE)
ax.set_ylabel('Ordering Energy (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(c) Ordering Energy Distribution', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.tick_params(axis='x', rotation=45)

# 3d: Min/Max/Mean formation energy by composition
ax = axes[1, 1]
comps_plot = comp_avg[comp_avg['x_V'].between(0.01, 0.99)]
x_vals = comps_plot['x_V'].values
ax.fill_between(x_vals, comps_plot['min_dHf']/1000, comps_plot['max_dHf']/1000, 
               alpha=0.2, color=PALETTE[0], label='Range (min-max)')
ax.plot(x_vals, comps_plot['mean_dHf']/1000, 'o-', color=PALETTE[0], 
       linewidth=2, markersize=8, label='Mean (converged)')
ax.plot(x_vals, comps_plot['min_dHf']/1000, 'v--', color=PALETTE[2], 
       linewidth=1.5, markersize=6, label='Most stable')
# RK prediction
ax.plot(x_plot, dHf_rk_plot/1000, 'k--', linewidth=2, label='RK-2 fit (A2)')
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(d) Energy Envelope vs RK Prediction', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(f'{OUT}/03_ordering_energy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 03_ordering_energy.png")

# --- Figure 4: B2 Order-Disorder Detail ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 4a: Fe4V4 configuration landscape
ax = axes[0, 0]
df_eq = df[df['n_v'] == 4].copy()
ax.scatter(df_eq['eta'], df_eq['sro_alpha'], c=df_eq['dH_f_Jmol_corrected']/1000,
          cmap='RdYlBu_r', s=60, alpha=0.7, edgecolors='gray', linewidth=0.5)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_xlabel(r'$\eta$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'SRO $\alpha_1$', fontsize=LABEL_SIZE)
ax.set_title(r'(a) Fe$_4$V$_4$ Configuration Space', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 4b: Sublattice occupation (y_V^A vs y_V^B)
ax = axes[0, 1]
scatter = ax.scatter(df_eq['y_V_A'], df_eq['y_V_B'], c=df_eq['dH_f_Jmol_corrected']/1000,
                    cmap='RdYlBu_r', s=60, alpha=0.7, edgecolors='gray', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Disordered')
ax.set_xlabel(r'$y_V^A$ (sublattice A)', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$y_V^B$ (sublattice B)', fontsize=LABEL_SIZE)
ax.set_title(r'(b) Fe$_4$V$_4$ Sublattice Occupation', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 4c: Energy vs eta for all equiatomic-like compositions
ax = axes[1, 0]
for nv in [3, 4, 5]:
    sub = df[df['n_v'] == nv]
    comp_label = f"Fe$_{{{8-nv}}}$V$_{{{nv}}}$"
    ax.scatter(sub['eta'], sub['dH_f_Jmol_corrected']/1000, label=comp_label,
              s=30, alpha=0.6)
ax.set_xlabel(r'$\eta$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(r'(c) $\eta$ vs $\Delta H_f$ (Near-equiatomic)', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 4d: Perfect B2 vs disordered energy comparison
ax = axes[1, 1]
ordered_energies = []
disordered_energies = []
compositions_x = []
for nv in range(1, 8):
    sub = df[df['n_v'] == nv]
    xv = nv / 8.0
    e_min = sub['dH_f_Jmol_corrected'].min()
    e_max = sub['dH_f_Jmol_corrected'].max()
    e_mean = sub['dH_f_Jmol_corrected'].mean()
    ordered_energies.append(e_min)
    disordered_energies.append(e_mean)
    compositions_x.append(xv)

ax.plot(compositions_x, np.array(ordered_energies)/1000, 'o-', color=PALETTE[2], 
       linewidth=2, markersize=8, label='Most ordered (min)')
ax.plot(compositions_x, np.array(disordered_energies)/1000, 's-', color=PALETTE[0],
       linewidth=2, markersize=8, label='Average (random-like)')
ax.fill_between(compositions_x, np.array(ordered_energies)/1000, 
               np.array(disordered_energies)/1000, alpha=0.15, color=PALETTE[3])
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(d) Ordered vs Disordered Energies', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# Add ordering energy annotation
for i, xv in enumerate(compositions_x):
    de = (disordered_energies[i] - ordered_energies[i])/1000
    if de > 3:
        ax.annotate(f'{de:.0f}', xy=(xv, (ordered_energies[i]+disordered_energies[i])/2000),
                   fontsize=ANNOT_SIZE, ha='center', color=PALETTE[3])

plt.tight_layout()
plt.savefig(f'{OUT}/04_b2_order_disorder.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 04_b2_order_disorder.png")

# --- Figure 5: Interaction Parameter Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 5a: RK fit to DFT data
ax = axes[0, 0]
x_plot_dense = np.linspace(0.001, 0.999, 500)
for n_terms, label, color in [(1, 'RK-1 (L$_0$)', PALETTE[0]),
                                (2, 'RK-2 (L$_0$, L$_1$)', PALETTE[1]),
                                (4, 'RK-4', PALETTE[3])]:
    y_rk = rk_basis(x_plot_dense, n_terms) @ rk_results[n_terms]['L']
    ax.plot(x_plot_dense, y_rk/1000, '-', color=color, linewidth=2, label=label)

# HKR parameters
y_hkr = x_plot_dense * (1-x_plot_dense) * (L0_HKR + L1_HKR*(1-2*x_plot_dense))
ax.plot(x_plot_dense, y_hkr/1000, 'k--', linewidth=2, label='HKR (1991)')

# DFT data (composition averages)
ax.scatter(comp_avg['x_V'], comp_avg['mean_dHf']/1000, c='red', s=80, zorder=5,
          edgecolors='black', linewidth=1, label='DFT mean')
ax.errorbar(comp_avg['x_V'], comp_avg['mean_dHf']/1000, 
           yerr=comp_avg['std_dHf']/1000, fmt='none', ecolor='red', alpha=0.5)

ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(a) Redlich-Kister Fits vs DFT', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE-1)
ax.tick_params(labelsize=TICK_SIZE)

# 5b: Residuals from RK-2 vs eta
ax = axes[0, 1]
df_mix2 = df[(df['n_v'] > 0) & (df['n_v'] < 8)].copy()
scatter = ax.scatter(df_mix2['eta'], df_mix2['dHf_deviation']/1000,
                    c=df_mix2['x_V'], cmap='viridis', s=30, alpha=0.6,
                    edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'$x_V$', fontsize=LABEL_SIZE)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\eta$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'Residual from RK-2 (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(r'(b) RK-2 Residual vs $\eta$ (Ordering Effect)', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 5c: Temperature-dependent phase stability schematic
ax = axes[1, 0]
T_range = np.linspace(300, 2500, 500)
# BCC_A2 at x_V=0.5: G_A2 = x*x * (L0 + L1*(1-2x)) at 0K baseline
# Add T-dependent terms
L0_T = -21427 + 6.846 * T_range
L1_T = 7345 - 1.509 * T_range
x_eq = 0.5
G_A2_mix = x_eq * (1-x_eq) * (L0_T + L1_T * (1-2*x_eq))
# Ideal mixing entropy
R = 8.314
S_ideal = -R * T_range * (x_eq * np.log(x_eq) + (1-x_eq) * np.log(1-x_eq))

# B2 ordering energy (estimated from DFT at 0K)
E_B2_ordering = df[df['n_v']==4]['dH_f_Jmol_corrected'].min() - df[df['n_v']==4]['dH_f_Jmol_corrected'].mean()
# Temperature dependence: ordering decreases with T
G_B2_ordering = E_B2_ordering * np.exp(-T_range / 1500)  # approximate

G_total_A2 = G_A2_mix + S_ideal
G_total_B2 = G_A2_mix + S_ideal + G_B2_ordering

ax.plot(T_range, G_A2_mix/1000, '-', color=PALETTE[0], linewidth=2, label=r'$G^{mix}_{A2}$ (RK)')
ax.plot(T_range, (G_A2_mix + G_B2_ordering)/1000, '-', color=PALETTE[2], linewidth=2, label=r'$G^{mix}_{A2} + G^{ord}_{B2}$')
ax.plot(T_range, G_B2_ordering/1000, '--', color=PALETTE[3], linewidth=2, label=r'$G^{ord}_{B2}$ (ordering)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Temperature (K)', fontsize=LABEL_SIZE)
ax.set_ylabel('Gibbs Energy (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(r'(c) Temperature Dependence at $x_V$ = 0.5', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 5d: L parameter comparison table
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Source', 'L$_0$ (J/mol)', 'L$_1$ (J/mol)', 'L$_2$', 'L$_3$'],
    ['HKR (1991)', f'{L0_HKR}', f'{L1_HKR}', '-', '-'],
    ['DFT (RK-2)', f'{rk_results[2]["L"][0]:.0f}', f'{rk_results[2]["L"][1]:.0f}', '-', '-'],
    ['DFT (RK-4)', f'{rk_results[4]["L"][0]:.0f}', f'{rk_results[4]["L"][1]:.0f}', 
     f'{rk_results[4]["L"][2]:.0f}', f'{rk_results[4]["L"][3]:.0f}'],
    ['', '', '', '', ''],
    ['Model', 'RMSE (kJ/mol)', 'R²', 'N$_{param}$', ''],
    ['RK-2 (A2)', f'{rk_results[2]["rmse"]/1000:.1f}', f'{rk_results[2]["r2"]:.4f}', '2', ''],
    ['2-SL B2', f'{rmse_2sl/1000:.1f}', f'{r2_2sl:.4f}', '4', ''],
    ['2-SL+L B2', f'{rmse_2sl_ext/1000:.1f}', f'{r2_2sl_ext:.4f}', '6', ''],
    ['8-SL CEF', '0.0', '1.0000', '256', ''],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(ANNOT_SIZE+1)
table.scale(1.2, 1.8)
# Style header rows
for j in range(5):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[5, j].set_facecolor('#4472C4')
    table[5, j].set_text_props(color='white', fontweight='bold')
ax.set_title('(d) Parameter Comparison Summary', fontsize=TITLE_SIZE, pad=20)

plt.tight_layout()
plt.savefig(f'{OUT}/05_interaction_parameters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 05_interaction_parameters.png")

# --- Figure 6: Degeneracy and Symmetry ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 6a: Energy level degeneracy
ax = axes[0]
ue_energies = [ue['energy']/1000 for ue in unique_energies]
ue_counts = [ue['count'] for ue in unique_energies]
ax.scatter(ue_energies, ue_counts, c=PALETTE[0], s=40, alpha=0.7, edgecolors='gray')
ax.set_xlabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel('Degeneracy (number of configs)', fontsize=LABEL_SIZE)
ax.set_title(f'(a) Energy Level Degeneracy ({len(unique_energies)} unique levels)', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 6b: Convex hull analysis
ax = axes[1]
# Plot all points
ax.scatter(df['x_V'], df['dH_f_Jmol_corrected']/1000, c='lightgray', s=20, alpha=0.5, label='All configs')
# Plot composition minima
min_per_comp = df.groupby('x_V')['dH_f_Jmol_corrected'].min().reset_index()
ax.plot(min_per_comp['x_V'], min_per_comp['dH_f_Jmol_corrected']/1000, 'o-', 
       color=PALETTE[2], linewidth=2, markersize=8, label='Ground state line')
# Convex hull
from scipy.spatial import ConvexHull
hull_points = np.column_stack([min_per_comp['x_V'], min_per_comp['dH_f_Jmol_corrected']/1000])
# Add endpoints
hull_points_ext = np.vstack([[0, 0], hull_points, [1, 0]])
# Lower convex hull
from scipy.interpolate import interp1d
hull_x = hull_points_ext[:, 0]
hull_y = hull_points_ext[:, 1]
# Find lower hull manually
lower_hull_x = [0]
lower_hull_y = [0]
for i in range(1, len(hull_x)-1):
    if hull_y[i] < 0:  # Only include points below reference line
        # Check if this point is on the lower hull
        if i == 0 or hull_y[i] <= np.interp(hull_x[i], [lower_hull_x[-1], hull_x[-1]], [lower_hull_y[-1], hull_y[-1]]):
            lower_hull_x.append(hull_x[i])
            lower_hull_y.append(hull_y[i])
lower_hull_x.append(1)
lower_hull_y.append(0)
ax.plot(lower_hull_x, lower_hull_y, 'r--', linewidth=2, label='Convex hull (approx)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\Delta H_f$ (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title('(b) Ground State Convex Hull', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(f'{OUT}/06_degeneracy_hull.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 06_degeneracy_hull.png")

# --- Figure 7: Old vs New Dataset Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Build comparison dataframe
old_values = []
new_values = []
func_names = []
for func in sorted(common_funcs):
    old_values.append(old_gf[func]/1000)
    new_values.append(new_gf[func]/1000)
    func_names.append(func)
old_values = np.array(old_values)
new_values = np.array(new_values)

# 7a: Parity plot
ax = axes[0]
ax.scatter(old_values, new_values, c=PALETTE[0], s=30, alpha=0.6, edgecolors='gray', linewidth=0.3)
lim = [min(old_values.min(), new_values.min()) - 20, max(old_values.max(), new_values.max()) + 20]
ax.plot(lim, lim, 'r--', linewidth=1.5)
ax.set_xlabel('Old TDB (Book1.xlsx) GF values (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel('New TDB (Fe-V_VASP.xlsx) GF values (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_title(f'(a) Old vs New GF Values ({len(common_funcs)} common)', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# 7b: Histogram of differences
ax = axes[1]
diffs_kj = (np.array([new_gf[f] for f in sorted(common_funcs)]) - 
            np.array([old_gf[f] for f in sorted(common_funcs)])) / 1000
ax.hist(diffs_kj, bins=30, color=PALETTE[0], alpha=0.7, edgecolor='gray')
ax.axvline(x=np.mean(diffs_kj), color='red', linewidth=2, label=f'Mean = {np.mean(diffs_kj):.1f} kJ/mol')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('New - Old (kJ/mol)', fontsize=LABEL_SIZE)
ax.set_ylabel('Count', fontsize=LABEL_SIZE)
ax.set_title('(b) Distribution of GF Differences', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(f'{OUT}/07_old_vs_new.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 07_old_vs_new.png")

# --- Figure 8: Metastable B2 Phase Diagram Schematic ---
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Schematic phase diagram showing A2/B2 transition
# At various temperatures, calculate the boundary between A2 and B2 stability
# Using the ordering energy estimate

T_arr = np.linspace(300, 2000, 200)
x_arr = np.linspace(0.01, 0.99, 200)

# For each composition, estimate the A2/B2 transition temperature
# B2 is stable when G_ord < -T*dS_config_ordered
# Simplified: T_trans ~ |E_ordering| / (k_B * ln(N_configs))

# Estimate ordering energy as function of composition
ordering_by_comp = {}
for nv in range(1, 8):
    sub = df[df['n_v'] == nv]
    xv = nv / 8.0
    E_avg = sub['dH_f_Jmol_corrected'].mean()
    E_min = sub['dH_f_Jmol_corrected'].min()
    ordering_by_comp[xv] = E_avg - E_min  # positive = B2 is more stable

# Interpolate ordering energy
xv_data = sorted(ordering_by_comp.keys())
oe_data = [ordering_by_comp[x] for x in xv_data]
oe_interp = interp1d(xv_data, oe_data, kind='quadratic', fill_value=0, bounds_error=False)

# Rough transition temperature estimate: T_trans ~ E_ordering / (R * delta_S_factor)
# delta_S_factor accounts for configurational entropy loss upon ordering
# For B2 at x=0.5: dS ~ R * ln(2) ≈ 5.76 J/mol/K
R_gas = 8.314
x_phase = np.linspace(0.05, 0.95, 100)
T_trans = oe_interp(x_phase) / (R_gas * 0.7)  # approximate

# Plot B2 region
ax.fill_between(x_phase, 0, T_trans, alpha=0.3, color=PALETTE[1], label='B2 (ordered)')
ax.fill_between(x_phase, T_trans, 2200, alpha=0.2, color=PALETTE[0], label='A2 (disordered)')
ax.plot(x_phase, T_trans, 'k-', linewidth=2.5, label='A2/B2 transition (estimated)')

# Mark experimental sigma phase region
ax.axvspan(0.3, 0.55, ymin=0, ymax=0.4, alpha=0.1, color='red', label=r'$\sigma$ phase (experimental)')

ax.set_xlabel(r'$x_V$', fontsize=LABEL_SIZE)
ax.set_ylabel('Temperature (K)', fontsize=LABEL_SIZE)
ax.set_title('Estimated Metastable BCC Phase Diagram (A2 + B2$_{221}$)', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 2200)
ax.tick_params(labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(f'{OUT}/08_metastable_phase_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 08_metastable_phase_diagram.png")

# ============================================================
# Save numerical results
# ============================================================
# Save RK parameters
rk_summary = pd.DataFrame({
    'n_terms': [1, 2, 3, 4],
    'L0': [rk_results[i]['L'][0] for i in [1,2,3,4]],
    'L1': [0] + [rk_results[i]['L'][1] for i in [2,3,4]],
    'L2': [0, 0] + [rk_results[i]['L'][2] for i in [3,4]],
    'L3': [0, 0, 0, rk_results[4]['L'][3]],
    'RMSE_Jmol': [rk_results[i]['rmse'] for i in [1,2,3,4]],
    'R2': [rk_results[i]['r2'] for i in [1,2,3,4]],
})
rk_summary.to_csv(f'{OUT}/rk_parameters.csv', index=False)

# Save SRO data
sro_summary = df[['config_index', 'composition', 'n_v', 'x_V', 'sro_alpha', 'eta', 
                   'y_V_A', 'y_V_B', 'dH_f_Jmol_corrected', 'dHf_deviation']].copy()
sro_summary.to_csv(f'{OUT}/sro_analysis.csv', index=False)

# Save ordering energies
ord_summary = []
for nv in range(1, 8):
    sub = df[df['n_v'] == nv]
    xv = nv / 8.0
    comp = sub['composition'].iloc[0]
    ord_summary.append({
        'composition': comp,
        'x_V': xv,
        'E_mean': sub['dH_f_Jmol_corrected'].mean(),
        'E_min': sub['dH_f_Jmol_corrected'].min(),
        'E_max': sub['dH_f_Jmol_corrected'].max(),
        'E_ordering': sub['dH_f_Jmol_corrected'].min() - sub['dH_f_Jmol_corrected'].mean(),
        'n_configs': len(sub),
        'most_stable_config': sub.loc[sub['dH_f_Jmol_corrected'].idxmin(), 'config_index'],
        'most_stable_eta': sub.loc[sub['dH_f_Jmol_corrected'].idxmin(), 'eta'],
    })
pd.DataFrame(ord_summary).to_csv(f'{OUT}/ordering_energies.csv', index=False)

print("\n" + "="*70)
print("Analysis complete! Files saved to:", OUT)
print("="*70)
print("\nFigures: 8 PNG files")
print("Data: rk_parameters.csv, sro_analysis.csv, ordering_energies.csv")
