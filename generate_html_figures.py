#!/usr/bin/env python3
"""
Generate all figures for the HTML student report.
Reads CSV data from hea_xgboost_output/ and data/ directories,
produces PNG files in html_figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -- Style --
rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 150

OUTDIR = os.path.join(os.path.dirname(__file__), 'html_figures')
os.makedirs(OUTDIR, exist_ok=True)

DATA = os.path.join(os.path.dirname(__file__), 'hea_xgboost_output')
RAWDATA = os.path.join(os.path.dirname(__file__), 'data')


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved: {path}")


# ================================================================
# Fig 1: Parity plot (predicted vs experimental)
# ================================================================
def fig_parity():
    df = pd.read_csv(os.path.join(DATA, 'detailed_predictions.csv'))
    fig, ax = plt.subplots(figsize=(7, 7))

    bcc = df[df['struct'] == 'BCC']
    fcc = df[df['struct'] == 'FCC']

    ax.scatter(bcc['a_exp'], bcc['a_eq10_ss'], c='#e74c3c', s=60, alpha=0.8,
               label=f'BCC ({len(bcc)})', edgecolors='white', linewidth=0.5, zorder=3)
    ax.scatter(fcc['a_exp'], fcc['a_eq10_ss'], c='#3498db', s=60, alpha=0.8,
               label=f'FCC ({len(fcc)})', edgecolors='white', linewidth=0.5, zorder=3)

    lo = min(df['a_exp'].min(), df['a_eq10_ss'].min()) - 0.05
    hi = max(df['a_exp'].max(), df['a_eq10_ss'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='完全一致線')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('実験格子定数 (Å)')
    ax.set_ylabel('予測格子定数 (Å)')
    ax.set_title('パリティプロット: DFT-Ωsf モデル')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    rmse = np.sqrt(np.mean((df['a_exp'] - df['a_eq10_ss'])**2))
    ax.text(0.05, 0.92, f'RMSE = {rmse:.4f} Å', transform=ax.transAxes,
            fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save(fig, 'fig_parity.png')


# ================================================================
# Fig 2: RMSE bar chart
# ================================================================
def fig_rmse_bar():
    stats = pd.read_csv(os.path.join(DATA, 'comparison_statistics.csv'))
    methods = {
        'Alonso Vegard': '単純平均\n(Alonso)',
        'Alonso Eq.10': 'Alonso\n体積ズレ補正',
        'King Vegard (this work)': '本研究\n単純平均',
        'DFT Eq.10 SS (this work)': '本研究\nDFT-Ωsf',
        'SS Eq.10 + Ridge': '物理+Ridge',
        'XGBoost LOO-CV': 'XGBoost\n(ML単独)',
    }
    sel = stats[stats['Method'].isin(methods.keys())].copy()
    sel['label'] = sel['Method'].map(methods)
    sel = sel.sort_values('RMSE_Ang', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#95a5a6' if v > 0.025 else '#e67e22' if v > 0.022
              else '#27ae60' for v in sel['RMSE_Ang']]
    bars = ax.barh(sel['label'], sel['RMSE_Ang'], color=colors, edgecolor='white')
    ax.set_xlabel('RMSE (Å)')
    ax.set_title('手法別 RMSE 比較 (64 HEA)')

    for bar, val in zip(bars, sel['RMSE_Ang']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=11)

    ax.axvline(x=0.0157, color='red', linestyle=':', alpha=0.7, label='ノイズフロア (0.0157)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(sel['RMSE_Ang']) * 1.3)
    ax.grid(axis='x', alpha=0.3)

    save(fig, 'fig_rmse_bar.png')


# ================================================================
# Fig 3: BCC vs FCC error distribution
# ================================================================
def fig_bcc_fcc():
    df = pd.read_csv(os.path.join(DATA, 'detailed_predictions.csv'))
    df['error'] = df['a_eq10_ss'] - df['a_exp']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, struct, color in zip(axes, ['BCC', 'FCC'], ['#e74c3c', '#3498db']):
        sub = df[df['struct'] == struct]
        ax.hist(sub['error'], bins=15, color=color, alpha=0.7, edgecolor='white')
        rmse = np.sqrt(np.mean(sub['error']**2))
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('予測誤差 (Å)')
        ax.set_ylabel('度数')
        ax.set_title(f'{struct} (N={len(sub)}, RMSE={rmse:.4f} Å)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('BCC/FCC 構造別の誤差分布', fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_bcc_fcc.png')


# ================================================================
# Fig 4: Independent test
# ================================================================
def fig_indep_test():
    df = pd.read_csv(os.path.join(DATA, 'independent_test_results.csv'))

    # Outliers to exclude from main evaluation
    outlier_compositions = ['Nb-Ti-V-Zr', 'Cr-Mo-Nb-Ta-V-W']
    is_outlier = df['composition'].isin(outlier_compositions)
    df_main = df[~is_outlier]
    df_outlier = df[is_outlier]

    fig, ax = plt.subplots(figsize=(7, 7))

    bcc_main = df_main[df_main['struct'] == 'BCC']
    fcc_main = df_main[df_main['struct'] == 'FCC']

    ax.scatter(bcc_main['a_exp'], bcc_main['a_eq10_ss'], c='#e74c3c', s=80, alpha=0.8,
               label=f'BCC ({len(bcc_main)})', edgecolors='black', linewidth=0.5, zorder=3)
    ax.scatter(fcc_main['a_exp'], fcc_main['a_eq10_ss'], c='#3498db', s=80, alpha=0.8,
               label=f'FCC ({len(fcc_main)})', edgecolors='black', linewidth=0.5, zorder=3)

    lo = min(df_main['a_exp'].min(), df_main['a_eq10_ss'].min()) - 0.05
    hi = max(df_main['a_exp'].max(), df_main['a_eq10_ss'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('実験格子定数 (Å)')
    ax.set_ylabel('予測格子定数 (Å)')
    ax.set_title('独立テスト 18 HEA（外れ値2点除外）')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    rmse_18 = np.sqrt(np.mean((df_main['a_exp'] - df_main['a_eq10_ss'])**2))
    ax.text(0.05, 0.92, f'RMSE = {rmse_18:.4f} Å (18 HEA)', transform=ax.transAxes,
            fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save(fig, 'fig_indep_test.png')


# ================================================================
# Fig 5: Element delta parameters
# ================================================================
def fig_element_delta():
    omega = pd.read_csv(os.path.join(DATA, 'omega_sf_data.csv'))
    # Compute additive decomposition: delta_i from mean omega per element
    elements = set()
    for pair in omega['pair']:
        a, b = pair.split('-')
        elements.add(a)
        elements.add(b)
    elements = sorted(elements)

    # Build element-mean omega
    el_omega = {el: [] for el in elements}
    for _, row in omega.iterrows():
        a, b = row['pair'].split('-')
        if pd.notna(row['omega_b2']):
            el_omega[a].append(row['omega_b2'])
            el_omega[b].append(row['omega_b2'])

    delta = {el: np.mean(vals) if vals else 0 for el, vals in el_omega.items()}
    delta_df = pd.DataFrame(list(delta.items()), columns=['element', 'delta'])
    delta_df = delta_df.sort_values('delta')

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in delta_df['delta']]
    ax.bar(range(len(delta_df)), delta_df['delta'], color=colors, edgecolor='white')
    ax.set_xticks(range(len(delta_df)))
    ax.set_xticklabels(delta_df['element'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('平均 Ωsf (B2)')
    ax.set_title('38元素の平均体積サイズファクター（正=膨張、負=収縮）')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    save(fig, 'fig_element_delta.png')


# ================================================================
# Fig 6: Additive fit quality
# ================================================================
def fig_additive_fit():
    omega = pd.read_csv(os.path.join(DATA, 'omega_sf_data.csv'))

    elements = set()
    for pair in omega['pair']:
        a, b = pair.split('-')
        elements.add(a)
        elements.add(b)
    elements = sorted(elements)

    el_omega = {el: [] for el in elements}
    for _, row in omega.iterrows():
        a, b = row['pair'].split('-')
        if pd.notna(row['omega_b2']):
            el_omega[a].append(row['omega_b2'])
            el_omega[b].append(row['omega_b2'])
    delta = {el: np.mean(vals) if vals else 0 for el, vals in el_omega.items()}

    actual, predicted = [], []
    for _, row in omega.iterrows():
        if pd.notna(row['omega_b2']):
            a, b = row['pair'].split('-')
            actual.append(row['omega_b2'])
            predicted.append(delta[a] + delta[b])

    actual = np.array(actual)
    predicted = np.array(predicted)
    r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.3, s=20, c='#2980b9', edgecolors='none')
    lo = min(actual.min(), predicted.min()) - 0.02
    hi = max(actual.max(), predicted.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, alpha=0.7)
    ax.set_xlabel('DFT計算 Ωsf')
    ax.set_ylabel('加法近似 (δi + δj)')
    ax.set_title('加法分解の品質')
    ax.text(0.05, 0.92, f'R² = {r2:.3f}\nN = {len(actual)} ペア',
            transform=ax.transAxes, fontsize=13,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    save(fig, 'fig_additive_fit.png')


# ================================================================
# Fig 7: delta_r structural invariance proof
# ================================================================
def fig_delta_r_proof():
    desc = pd.read_csv(os.path.join(DATA, 'descriptor_analysis.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bcc = desc[desc['struct'] == 'BCC']
    fcc = desc[desc['struct'] == 'FCC']

    # (a) B2 lattice vs delta_r
    axes[0].scatter(bcc['delta_r'], bcc['a_eq10_ss'], c='#e74c3c', s=40, alpha=0.7)
    axes[0].set_xlabel('δr (%)')
    axes[0].set_ylabel('B2 格子定数 (Å)')
    axes[0].set_title('(a) BCC: δr vs 格子定数')
    axes[0].grid(True, alpha=0.3)

    # (b) L12 lattice vs delta_r
    axes[1].scatter(fcc['delta_r'], fcc['a_eq10_ss'], c='#3498db', s=40, alpha=0.7)
    axes[1].set_xlabel('δr (%)')
    axes[1].set_ylabel('L1₂ 格子定数 (Å)')
    axes[1].set_title('(b) FCC: δr vs 格子定数')
    axes[1].grid(True, alpha=0.3)

    # (c) Error vs delta_r (both)
    axes[2].scatter(bcc['delta_r'], bcc['a_eq10_ss'] - bcc['a_exp'],
                    c='#e74c3c', s=40, alpha=0.7, label='BCC')
    axes[2].scatter(fcc['delta_r'], fcc['a_eq10_ss'] - fcc['a_exp'],
                    c='#3498db', s=40, alpha=0.7, label='FCC')
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('δr (%)')
    axes[2].set_ylabel('予測誤差 (Å)')
    axes[2].set_title('(c) δr vs 予測誤差')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('δr（サイズミスマッチ）は構造情報を含まない', fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_delta_r_proof.png')


# ================================================================
# Fig 8: ROC curve for phase classification
# ================================================================
def fig_roc():
    mc = pd.read_csv(os.path.join(DATA, 'multiphase_classification.csv'))
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 7))

    for feat, label, color in [
        ('delta_r', 'δr', '#e74c3c'),
        ('delta_sf_combined', 'δΩsf (combined)', '#3498db'),
        ('omega_yz', 'Ω (Yang-Zhang)', '#27ae60'),
        ('VEC', 'VEC', '#f39c12'),
    ]:
        if feat in mc.columns:
            y_true = mc['is_ss'].values
            scores = mc[feat].values
            mask = ~np.isnan(scores)
            if mask.sum() < 5:
                continue
            fpr, tpr, _ = roc_curve(y_true[mask], -scores[mask])
            roc_auc = auc(fpr, tpr)
            if roc_auc < 0.5:
                fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
                roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('偽陽性率 (FPR)')
    ax.set_ylabel('真陽性率 (TPR)')
    ax.set_title('ROC曲線: 固溶体 vs 多相の分類')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    save(fig, 'fig_roc.png')


# ================================================================
# Fig 9: Phase stability map
# ================================================================
def fig_phase_map():
    mc = pd.read_csv(os.path.join(DATA, 'multiphase_classification.csv'))

    fig, ax = plt.subplots(figsize=(8, 6))

    ss = mc[mc['is_ss'] == 1]
    mp = mc[mc['is_ss'] == 0]

    ax.scatter(ss['delta_r'], ss['omega_yz'], c='#27ae60', s=60, alpha=0.7,
               label=f'固溶体 ({len(ss)})', edgecolors='white', linewidth=0.5)
    ax.scatter(mp['delta_r'], mp['omega_yz'], c='#e74c3c', s=60, alpha=0.7,
               label=f'多相/金属間 ({len(mp)})', edgecolors='white', linewidth=0.5)

    ax.set_xlabel('δr (%)')
    ax.set_ylabel('Ω (Yang-Zhang)')
    ax.set_title('相安定性マップ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, 'fig_phase_map.png')


# ================================================================
# Fig A1: Composition-dependent V_eff examples
# ================================================================
def fig_composition_examples():
    omega = pd.read_csv(os.path.join(DATA, 'omega_sf_data.csv'))
    # Pick interesting pairs with large omega
    omega_b2 = omega.dropna(subset=['omega_b2']).copy()
    omega_b2['abs_omega'] = omega_b2['omega_b2'].abs()
    top_pairs = omega_b2.nlargest(6, 'abs_omega')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        ax = axes[idx]
        pair = row['pair']
        omega_val = row['omega_b2']
        a, b = pair.split('-')

        x = np.linspace(0, 1, 50)
        vegard = x  # normalized
        corrected = x * (1 + (1-x) * omega_val)

        ax.plot(x, vegard, 'b--', label='Vegard (直線)', alpha=0.7)
        ax.plot(x, corrected, 'r-', label=f'Ωsf補正 ({omega_val:+.3f})', lw=2)
        ax.fill_between(x, vegard, corrected, alpha=0.2, color='red')
        ax.set_xlabel(f'{b} 組成比')
        ax.set_ylabel('正規化体積')
        ax.set_title(f'{a}-{b}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('代表的なペアの組成依存性: Vegardからのズレ', fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, 'fig_composition_examples.png')


# ================================================================
# Fig A3: L12 asymmetry
# ================================================================
def fig_l12_asymmetry():
    omega = pd.read_csv(os.path.join(DATA, 'omega_sf_data.csv'))
    b2 = omega.dropna(subset=['omega_b2'])
    l12 = omega.dropna(subset=['omega_l12'])

    # Compare B2 vs L12 for overlapping pairs
    merged = omega.dropna(subset=['omega_b2', 'omega_l12'])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged['omega_b2'], merged['omega_l12'], alpha=0.4, s=20,
               c='#8e44ad', edgecolors='none')
    lo = min(merged['omega_b2'].min(), merged['omega_l12'].min()) - 0.02
    hi = max(merged['omega_b2'].max(), merged['omega_l12'].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, alpha=0.7)
    ax.set_xlabel('Ωsf (B2構造)')
    ax.set_ylabel('Ωsf (L1₂構造)')
    ax.set_title('B2 vs L1₂ の構造非対称性')
    r = np.corrcoef(merged['omega_b2'], merged['omega_l12'])[0, 1]
    ax.text(0.05, 0.92, f'r = {r:.3f}\nN = {len(merged)} ペア',
            transform=ax.transAxes, fontsize=13,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    save(fig, 'fig_l12_asymmetry.png')


# ================================================================
# Fig: Noise floor visualization
# ================================================================
def fig_noise_floor():
    df = pd.read_csv(os.path.join(DATA, 'detailed_predictions.csv'))

    methods = {
        'a_vegard_alonso': 'Alonso単純平均',
        'a_eq10_alonso': 'Alonso体積ズレ補正',
        'a_eq10_ss': '本研究DFT-Ωsf',
        'a_ss_ridge': '物理+Ridge',
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    rmses = []
    labels = []
    for col, label in methods.items():
        if col in df.columns:
            rmse = np.sqrt(np.mean((df['a_exp'] - df[col])**2))
            rmses.append(rmse)
            labels.append(label)

    ax.bar(range(len(rmses)), rmses, color=['#95a5a6', '#e67e22', '#27ae60', '#2980b9'],
           edgecolor='white')
    ax.axhline(y=0.0157, color='red', linestyle=':', lw=2,
               label='ノイズフロア σ=0.0157 Å')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('RMSE (Å)')
    ax.set_title('各手法のRMSEとノイズフロア')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(rmses):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=11)

    save(fig, 'fig_noise_floor.png')


# ================================================================
# Fig: DFT self-consistent q~1
# ================================================================
def fig_dft_selfconsistent():
    fig, ax = plt.subplots(figsize=(8, 5))

    data = {
        'King参照\n(実験体積)': {'q_bcc': 0.49, 'q_fcc': 0.13, 'rmse': 0.0214},
        'VASP参照\n(DFT体積)': {'q_bcc': 1.17, 'q_fcc': 0.94, 'rmse': 0.0202},
        'VASP参照\nq=1固定': {'q_bcc': 1.00, 'q_fcc': 1.00, 'rmse': 0.0203},
    }

    x = np.arange(len(data))
    width = 0.25

    q_bcc = [d['q_bcc'] for d in data.values()]
    q_fcc = [d['q_fcc'] for d in data.values()]
    rmse = [d['rmse'] for d in data.values()]

    ax.bar(x - width, q_bcc, width, label='q_BCC', color='#e74c3c', alpha=0.8)
    ax.bar(x, q_fcc, width, label='q_FCC', color='#3498db', alpha=0.8)
    ax.bar(x + width, [r * 100 for r in rmse], width, label='RMSE×100',
           color='#27ae60', alpha=0.8)

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='q=1 (理想)')
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys())
    ax.set_ylabel('値')
    ax.set_title('参照体積の選択が q と精度に与える影響')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    save(fig, 'fig_dft_selfconsistent.png')


# ================================================================
# Fig: SQS convergence issues
# ================================================================
def fig_sqs_issues():
    fig, ax = plt.subplots(figsize=(8, 5))

    structures = ['B2\n(2原子/セル)', 'L1₂\n(4原子/セル)', 'SQS\n(16原子/セル)']
    convergence = [97.2, 90.1, 32.6]
    colors = ['#27ae60', '#f39c12', '#e74c3c']

    bars = ax.bar(structures, convergence, color=colors, edgecolor='white', width=0.5)
    ax.set_ylabel('収束率 (%)')
    ax.set_title('構造タイプ別のVASP収束率')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, convergence):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
                ha='center', fontsize=14, fontweight='bold')

    save(fig, 'fig_sqs_issues.png')


# ================================================================
# Fig A2: Hard sphere packing limit
# ================================================================
def fig_packing():
    """Show that hard-sphere contact model fails to reproduce volume deviations."""
    b2 = pd.read_csv(os.path.join(RAWDATA, 'compounds_VASP_B2.csv'))
    # Get pure element volumes (A=B cases)
    pure = b2[b2['element_A'] == b2['element_B']].copy()
    pure['v_atom'] = pure['lattice_constant']**3 / 2  # BCC: 2 atoms/cell
    v_pure = dict(zip(pure['element_A'], pure['v_atom']))

    # Get alloy volumes
    alloy = b2[b2['element_A'] != b2['element_B']].copy()
    alloy['v_atom'] = alloy['lattice_constant']**3 / 2

    vegard_vals, dft_vals = [], []
    for _, row in alloy.iterrows():
        a, b_el = row['element_A'], row['element_B']
        if a in v_pure and b_el in v_pure:
            v_veg = (v_pure[a] + v_pure[b_el]) / 2
            vegard_vals.append(v_veg)
            dft_vals.append(row['v_atom'])

    vegard_vals = np.array(vegard_vals)
    dft_vals = np.array(dft_vals)
    deviation = (dft_vals - vegard_vals) / vegard_vals * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) Vegard vs DFT
    axes[0].scatter(vegard_vals, dft_vals, alpha=0.2, s=10, c='#2980b9', edgecolors='none')
    lo = min(vegard_vals.min(), dft_vals.min()) - 1
    hi = max(vegard_vals.max(), dft_vals.max()) + 1
    axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1, alpha=0.7)
    axes[0].set_xlabel('Vegard体積 (Å³)')
    axes[0].set_ylabel('DFT体積 (Å³)')
    axes[0].set_title('(a) Vegard予測 vs DFT計算')
    axes[0].grid(True, alpha=0.3)

    # (b) Deviation histogram
    axes[1].hist(deviation, bins=50, color='#8e44ad', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Vegardからの偏差 (%)')
    axes[1].set_ylabel('度数')
    axes[1].set_title(f'(b) 体積偏差の分布 (N={len(deviation)})')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('剛体球モデルの限界: 実際の原子は「柔らかい球」', fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, 'fig_packing.png')


# ================================================================
# Fig A4: Volume vs radius
# ================================================================
def fig_volume_radius():
    """DFT atomic volume vs various radius definitions."""
    b2 = pd.read_csv(os.path.join(RAWDATA, 'compounds_VASP_B2.csv'))
    pure = b2[b2['element_A'] == b2['element_B']].copy()
    pure['v_atom'] = pure['lattice_constant']**3 / 2
    pure['r_eff'] = (3 * pure['v_atom'] / (4 * np.pi))**(1/3)

    # King radii (selected common elements)
    king_radii = {
        'Al': 1.582, 'Ti': 1.615, 'V': 1.489, 'Cr': 1.423, 'Mn': 1.428,
        'Fe': 1.411, 'Co': 1.385, 'Ni': 1.377, 'Cu': 1.413, 'Zn': 1.536,
        'Zr': 1.771, 'Nb': 1.625, 'Mo': 1.552, 'Pd': 1.521, 'Ag': 1.595,
        'Hf': 1.744, 'Ta': 1.625, 'W': 1.556, 'Pt': 1.534, 'Au': 1.593,
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    el_list = []
    for _, row in pure.iterrows():
        el = row['element_A']
        if el in king_radii:
            el_list.append({'element': el, 'r_king': king_radii[el],
                           'r_eff': row['r_eff'], 'v_atom': row['v_atom']})

    if el_list:
        df_el = pd.DataFrame(el_list)
        ax.scatter(df_el['r_king'], df_el['r_eff'], c='#e74c3c', s=80,
                   alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
        for _, row in df_el.iterrows():
            ax.annotate(row['element'], (row['r_king'], row['r_eff']),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)

        lo = min(df_el['r_king'].min(), df_el['r_eff'].min()) - 0.05
        hi = max(df_el['r_king'].max(), df_el['r_eff'].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        r = np.corrcoef(df_el['r_king'], df_el['r_eff'])[0, 1]
        ax.text(0.05, 0.92, f'r = {r:.3f}', transform=ax.transAxes, fontsize=13,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('King半径 (Å)')
    ax.set_ylabel('DFT有効半径 (Å)')
    ax.set_title('King実験半径 vs DFT有効半径')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    save(fig, 'fig_volume_radius.png')


# ================================================================
# Fig A5: How much structure effect is absorbed by Vegard
# ================================================================
def fig_vegard_structure_absorbed():
    """Show B2 vs L12 lattice constants and how Vegard treats them identically."""
    b2 = pd.read_csv(os.path.join(RAWDATA, 'compounds_VASP_B2.csv'))
    l12 = pd.read_csv(os.path.join(RAWDATA, 'compounds_VASP_L12.csv'))

    # Find overlapping pairs
    b2_pairs = {}
    for _, row in b2.iterrows():
        pair = tuple(sorted([row['element_A'], row['element_B']]))
        if row['element_A'] != row['element_B']:
            b2_pairs[pair] = row['lattice_constant']

    l12_pairs = {}
    for _, row in l12.iterrows():
        pair = tuple(sorted([row['element_A'], row['element_B']]))
        if row['element_A'] != row['element_B']:
            if pair not in l12_pairs:
                l12_pairs[pair] = []
            l12_pairs[pair].append(row['lattice_constant'])

    # Average L12 (A3B and B3A)
    common_pairs = set(b2_pairs.keys()) & set(l12_pairs.keys())

    a_b2, a_l12 = [], []
    for pair in common_pairs:
        a_b2.append(b2_pairs[pair])
        a_l12.append(np.mean(l12_pairs[pair]))

    a_b2 = np.array(a_b2)
    a_l12 = np.array(a_l12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) B2 vs L12 lattice constants
    axes[0].scatter(a_b2, a_l12, alpha=0.3, s=15, c='#2980b9', edgecolors='none')
    lo = min(a_b2.min(), a_l12.min()) - 0.1
    hi = max(a_b2.max(), a_l12.max()) + 0.1
    axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1, alpha=0.7)
    axes[0].set_xlabel('B2 格子定数 (Å)')
    axes[0].set_ylabel('L1₂ 格子定数 (Å)')
    axes[0].set_title(f'(a) B2 vs L1₂ (N={len(a_b2)} ペア)')
    r = np.corrcoef(a_b2, a_l12)[0, 1]
    axes[0].text(0.05, 0.92, f'r = {r:.3f}', transform=axes[0].transAxes, fontsize=13,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].grid(True, alpha=0.3)

    # (b) Difference histogram
    diff = a_b2 - a_l12
    axes[1].hist(diff, bins=50, color='#e67e22', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('B2 - L1₂ 格子定数差 (Å)')
    axes[1].set_ylabel('度数')
    axes[1].set_title(f'(b) 構造間差の分布 (平均={np.mean(diff):.3f} Å)')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Vegard則は構造効果を無視する', fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, 'fig_vegard_structure_absorbed.png')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("Generating HTML report figures...")

    fig_parity()
    fig_rmse_bar()
    fig_bcc_fcc()
    fig_indep_test()
    fig_element_delta()
    fig_additive_fit()
    fig_delta_r_proof()

    try:
        fig_roc()
    except ImportError:
        print("  skipped fig_roc (sklearn not available)")

    fig_phase_map()
    fig_composition_examples()
    fig_l12_asymmetry()
    fig_noise_floor()
    fig_dft_selfconsistent()
    fig_sqs_issues()
    fig_packing()
    fig_volume_radius()
    fig_vegard_structure_absorbed()

    print(f"\nDone! All figures saved to {OUTDIR}/")
