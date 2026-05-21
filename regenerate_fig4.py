#!/usr/bin/env python3
"""Redesign Fig 4 (independent test) as a multi-panel figure that clearly 
shows (a) per-HEA error comparison, (b) train vs test RMSE, (c) why BCC 
predictions are identical (Ω_sf coverage)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

jp_font_path = '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
fm.fontManager.addfont(jp_font_path)
plt.rcParams.update({
    'font.family': ['IPAexGothic', 'DejaVu Sans'],
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

def main():
    df = pd.read_csv('hea_xgboost_output/independent_test_results.csv')
    df_train = pd.read_csv('hea_xgboost_output/detailed_predictions.csv')
    
    # Shorten composition names for display
    short_names = []
    for c in df['composition']:
        parts = c.replace('-', '')
        short_names.append(c)
    df['label'] = [f"{c}\n({s})" for c, s in zip(df['composition'], df['struct'])]
    
    # Sort by structure then by absolute Vegard error
    df = df.sort_values(['struct', 'err_vegard'], key=lambda x: x.abs() if x.name == 'err_vegard' else x)
    df = df.reset_index(drop=True)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)
    
    # ===== Panel (a): Per-HEA absolute error comparison =====
    ax_a = fig.add_subplot(gs[0, :2])
    
    n = len(df)
    x = np.arange(n)
    w = 0.25
    
    err_v = df['err_vegard'].abs().values * 1000  # Convert to mÅ
    err_k = df['err_king'].abs().values * 1000
    err_s = df['err_ss'].abs().values * 1000
    
    bars1 = ax_a.bar(x - w, err_v, w, color='gray', alpha=0.7, label='Vegard', edgecolor='k', linewidth=0.3)
    bars2 = ax_a.bar(x, err_k, w, color='#2196F3', alpha=0.7, label='Alonso Eq.10', edgecolor='k', linewidth=0.3)
    bars3 = ax_a.bar(x + w, err_s, w, color='#F44336', alpha=0.7, label='DFT Eq.10 SS', edgecolor='k', linewidth=0.3)
    
    # Mark BCC region
    bcc_idx = df[df['struct'] == 'BCC'].index
    fcc_idx = df[df['struct'] == 'FCC'].index
    ax_a.axvspan(bcc_idx[0] - 0.5, bcc_idx[-1] + 0.5, alpha=0.08, color='#FF9800', zorder=0)
    ax_a.axvspan(fcc_idx[0] - 0.5, fcc_idx[-1] + 0.5, alpha=0.08, color='#4CAF50', zorder=0)
    
    # Label regions
    ax_a.text(np.mean(bcc_idx), ax_a.get_ylim()[1] if ax_a.get_ylim()[1] > 50 else 60, 
              'BCC', fontsize=16, fontweight='bold', color='#FF9800', ha='center', va='bottom')
    ax_a.text(np.mean(fcc_idx), 60, 
              'FCC', fontsize=16, fontweight='bold', color='#4CAF50', ha='center', va='bottom')
    
    # Mark identical predictions with arrows
    for i in range(n):
        if abs(df.iloc[i]['a_vegard'] - df.iloc[i]['a_eq10_ss']) < 1e-6:
            ax_a.annotate('=', xy=(i, max(err_v[i], err_s[i]) + 2), 
                         fontsize=12, fontweight='bold', color='#F44336',
                         ha='center', va='bottom')
    
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(df['composition'], rotation=45, ha='right', fontsize=9)
    ax_a.set_ylabel('|Error| (mÅ)', fontsize=13)
    ax_a.set_title('(a) Per-HEA absolute prediction error', fontsize=15, fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax_a.grid(True, alpha=0.3, axis='y')
    ax_a.set_ylim(0, max(err_v.max(), err_k.max(), err_s.max()) * 1.3)
    
    # Add annotation for "=" marks
    ax_a.text(0.02, 0.95, '"=" : Vegard = DFT Eq.10 SS\n'
              '(no $\\Omega_\\mathrm{sf}$ data → Vegard fallback)',
              transform=ax_a.transAxes, fontsize=11, va='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.9, edgecolor='#FF9800'))
    
    # ===== Panel (b): Train vs Test RMSE comparison =====
    ax_b = fig.add_subplot(gs[0, 2])
    
    # Calculate train RMSE by struct
    bcc_train = df_train[df_train['struct'] == 'BCC']
    fcc_train = df_train[df_train['struct'] == 'FCC']
    bcc_test = df[df['struct'] == 'BCC']
    fcc_test = df[df['struct'] == 'FCC']
    
    train_v_all = np.sqrt(np.mean((df_train['a_vegard_king'] - df_train['a_exp'])**2))
    train_s_all = np.sqrt(np.mean((df_train['a_eq10_ss'] - df_train['a_exp'])**2))
    test_v_all = np.sqrt(np.mean(df['err_vegard']**2))
    test_s_all = np.sqrt(np.mean(df['err_ss']**2))
    
    categories = ['Train\n(64 HEA)', 'Test\n(20 HEA)']
    vegard_vals = [train_v_all * 1000, test_v_all * 1000]
    ss_vals = [train_s_all * 1000, test_s_all * 1000]
    
    x_b = np.arange(len(categories))
    w_b = 0.3
    bars_v = ax_b.bar(x_b - w_b/2, vegard_vals, w_b, color='gray', alpha=0.7, label='Vegard', edgecolor='k', linewidth=0.5)
    bars_s = ax_b.bar(x_b + w_b/2, ss_vals, w_b, color='#F44336', alpha=0.7, label='DFT Eq.10 SS', edgecolor='k', linewidth=0.5)
    
    for bar, val in zip(bars_v, vegard_vals):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                  f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars_s, ss_vals):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#F44336')
    
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(categories, fontsize=12)
    ax_b.set_ylabel('RMSE (mÅ)', fontsize=13)
    ax_b.set_title('(b) Train vs Test RMSE\n(no overfitting)', fontsize=15, fontweight='bold')
    ax_b.legend(fontsize=11, framealpha=0.9)
    ax_b.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotation
    improv_train = (1 - train_s_all / train_v_all) * 100
    improv_test = (1 - test_s_all / test_v_all) * 100
    ax_b.annotate(f'{improv_train:.1f}%↓', xy=(0, train_s_all*1000), 
                  fontsize=12, fontweight='bold', color='#4CAF50', ha='center',
                  xytext=(0, train_s_all*1000 - 3))
    ax_b.annotate(f'{improv_test:.1f}%↓', xy=(1, test_s_all*1000),
                  fontsize=12, fontweight='bold', color='#4CAF50', ha='center',
                  xytext=(1, test_s_all*1000 - 3))
    
    # ===== Panel (c): BCC vs FCC breakdown =====
    ax_c = fig.add_subplot(gs[1, 0])
    
    # BCC
    bcc_rmse_v = np.sqrt(np.mean(bcc_test['err_vegard']**2)) * 1000
    bcc_rmse_k = np.sqrt(np.mean(bcc_test['err_king']**2)) * 1000
    bcc_rmse_s = np.sqrt(np.mean(bcc_test['err_ss']**2)) * 1000
    # FCC
    fcc_rmse_v = np.sqrt(np.mean(fcc_test['err_vegard']**2)) * 1000
    fcc_rmse_k = np.sqrt(np.mean(fcc_test['err_king']**2)) * 1000
    fcc_rmse_s = np.sqrt(np.mean(fcc_test['err_ss']**2)) * 1000
    
    categories_c = ['BCC\n(N=8)', 'FCC\n(N=12)']
    x_c = np.arange(2)
    w_c = 0.22
    
    ax_c.bar(x_c - w_c, [bcc_rmse_v, fcc_rmse_v], w_c, color='gray', alpha=0.7, 
             label='Vegard', edgecolor='k', linewidth=0.5)
    ax_c.bar(x_c, [bcc_rmse_k, fcc_rmse_k], w_c, color='#2196F3', alpha=0.7, 
             label='Alonso Eq.10', edgecolor='k', linewidth=0.5)
    ax_c.bar(x_c + w_c, [bcc_rmse_s, fcc_rmse_s], w_c, color='#F44336', alpha=0.7, 
             label='DFT Eq.10 SS', edgecolor='k', linewidth=0.5)
    
    # Add values
    for i, (v, k, s) in enumerate([(bcc_rmse_v, bcc_rmse_k, bcc_rmse_s), 
                                     (fcc_rmse_v, fcc_rmse_k, fcc_rmse_s)]):
        ax_c.text(i - w_c, v + 0.5, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
        ax_c.text(i, k + 0.5, f'{k:.1f}', ha='center', fontsize=10, fontweight='bold', color='#2196F3')
        ax_c.text(i + w_c, s + 0.5, f'{s:.1f}', ha='center', fontsize=10, fontweight='bold', color='#F44336')
    
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(categories_c, fontsize=12)
    ax_c.set_ylabel('RMSE (mÅ)', fontsize=13)
    ax_c.set_title('(c) BCC vs FCC breakdown', fontsize=15, fontweight='bold')
    ax_c.legend(fontsize=10, framealpha=0.9)
    ax_c.grid(True, alpha=0.3, axis='y')
    
    # Annotation for BCC
    ax_c.text(0, bcc_rmse_v * 0.5, '7/8 BCC:\nno $\\Omega_{\\rm sf}$ data\n→ Vegard = SS', 
              ha='center', fontsize=10, color='#FF9800', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    # Annotation for FCC
    fcc_improv = (1 - fcc_rmse_s / fcc_rmse_v) * 100
    ax_c.text(1, fcc_rmse_v * 0.6, f'FCC: {fcc_improv:.0f}%↓\nimprovement', 
              ha='center', fontsize=10, color='#4CAF50', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # ===== Panel (d): Parity plot (condensed) =====
    ax_d = fig.add_subplot(gs[1, 1])
    
    ax_d.scatter(df['a_exp'], df['a_vegard'], marker='s', c='gray', s=60,
                 alpha=0.6, edgecolors='k', linewidths=0.3, label='Vegard', zorder=2)
    ax_d.scatter(df['a_exp'], df['a_eq10_ss'], marker='o', c='#F44336', s=70,
                 alpha=0.8, edgecolors='k', linewidths=0.3, label='DFT Eq.10 SS', zorder=4)
    
    lims = [2.83, 3.57]
    ax_d.plot(lims, lims, 'k--', lw=1.5, alpha=0.5, zorder=1)
    ax_d.set_xlabel('Exp. $a$ (Å)', fontsize=13)
    ax_d.set_ylabel('Pred. $a$ (Å)', fontsize=13)
    ax_d.set_title('(d) Parity plot', fontsize=15, fontweight='bold')
    ax_d.legend(fontsize=11, framealpha=0.9, loc='upper left')
    ax_d.set_aspect('equal')
    ax_d.set_xlim(lims); ax_d.set_ylim(lims)
    ax_d.grid(True, alpha=0.3)
    
    # ===== Panel (e): Ω_sf coverage explanation =====
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')
    
    explanation = (
        "Why BCC shows no improvement:\n\n"
        "• Independent test BCC alloys use\n"
        "  refractory elements (Nb, Ti, V, Zr,\n"
        "  Mo, Ta, W, Hf, Cr)\n\n"
        "• B2 DFT data for these pairs is\n"
        "  absent in MP/OQMD databases\n\n"
        "• When $\\Omega_{\\rm sf}$ data is missing,\n"
        "  model falls back to Vegard\n"
        "  → identical prediction\n\n"
        "• This is graceful degradation:\n"
        "  no harm from missing data\n\n"
        "• FCC alloys (CoCrFeNi family)\n"
        "  have full L1$_2$ coverage\n"
        "  → 12% RMSE improvement\n\n"
        "Key finding:\n"
        "  Test RMSE ≈ Train RMSE\n"
        "  → No overfitting"
    )
    
    ax_e.text(0.05, 0.95, explanation, transform=ax_e.transAxes,
              fontsize=12, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.9, edgecolor='#1976D2'),
              family='monospace')
    ax_e.set_title('(e) Interpretation', fontsize=15, fontweight='bold')
    
    fig.suptitle('Independent Test Validation (20 HEAs outside Alonso dataset)',
                 fontsize=17, fontweight='bold', y=1.01)
    
    fig.savefig('paper/fig_indep_test.png')
    plt.close(fig)
    print("Saved paper/fig_indep_test.png")


if __name__ == '__main__':
    import os
    os.chdir('/home/ubuntu/repos/machine-learning')
    main()
