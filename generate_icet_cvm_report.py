#!/usr/bin/env python3
"""
icet CVM計算レポート生成スクリプト

このスクリプトは、icet_cvm_calculation.pyの実行結果を可視化し、
詳細な計算レポートをMarkdown形式で生成します。

使用方法:
    python generate_icet_cvm_report.py

出力:
    - docs/assets/icet_cvm_report/*.png: 各種可視化図
    - docs/icet_cvm_calculation_report_ja.md: 詳細レポート（日本語）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# 日本語フォント設定（利用可能な場合）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 出力ディレクトリ
ASSETS_DIR = "docs/assets/icet_cvm_report"
REPORT_FILE = "docs/icet_cvm_calculation_report_ja.md"

# グローバル変数（レポート生成時に設定）
REPORT_DATA = {}


def setup_directories():
    """出力ディレクトリを作成"""
    os.makedirs(ASSETS_DIR, exist_ok=True)


def run_calculation():
    """
    icet計算を実行し、結果を取得する
    """
    from ase.db import connect
    from ase.build import bulk, make_supercell
    from icet import ClusterSpace, StructureContainer
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import cross_val_score
    import random
    
    # 設定パラメータ
    DB_FILENAME = "fe_v_data.db"
    LATTICE_CONST = 2.87
    CUTOFFS = [6.0, 4.0, 4.0]
    N_STRUCTURES = 50
    
    # 乱数シード設定
    random.seed(42)
    np.random.seed(42)
    
    # ダミー構造の生成
    print("ダミー構造を生成中...")
    prim = bulk('Fe', 'bcc', a=LATTICE_CONST)
    
    supercell_sizes = [
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[2, 0, 0], [0, 2, 0], [0, 0, 3]],
    ]
    
    structures = []
    concentrations_list = []
    mixing_energies_list = []
    
    for _ in range(N_STRUCTURES):
        sc_matrix = random.choice(supercell_sizes)
        supercell = make_supercell(prim, sc_matrix)
        n_atoms = len(supercell)
        
        conc_v = random.uniform(0.0, 1.0)
        n_v = int(round(conc_v * n_atoms))
        n_fe = n_atoms - n_v
        
        symbols = ['Fe'] * n_fe + ['V'] * n_v
        random.shuffle(symbols)
        supercell.set_chemical_symbols(symbols)
        
        # 正規溶体モデルで混合エネルギーを計算
        c_v = n_v / n_atoms
        c_fe = 1.0 - c_v
        omega = 0.15
        asymmetric = 0.02 * c_fe * c_v * (c_fe - c_v)
        noise = np.random.normal(0, 0.003)
        mixing_energy = omega * c_fe * c_v + asymmetric + noise
        
        structures.append(supercell)
        concentrations_list.append(c_v)
        mixing_energies_list.append(mixing_energy)
    
    # データベース作成
    if os.path.exists(DB_FILENAME):
        os.remove(DB_FILENAME)
    
    db = connect(DB_FILENAME)
    for atoms, mixing_energy, conc_v in zip(structures, mixing_energies_list, concentrations_list):
        db.write(atoms, key_value_pairs={
            'mixing_energy': mixing_energy,
            'concentration_v': conc_v
        })
    
    # ClusterSpace構築
    print("ClusterSpaceを構築中...")
    cs = ClusterSpace(
        structure=prim,
        cutoffs=CUTOFFS,
        chemical_symbols=['Fe', 'V']
    )
    
    # StructureContainer作成
    sc = StructureContainer(cluster_space=cs)
    for row in db.select():
        sc.add_structure(
            structure=row.toatoms(),
            user_tag=str(row.id),
            properties={'mixing_energy': row.mixing_energy}
        )
    
    # フィッティング
    print("フィッティング実行中...")
    X, y = sc.get_fit_data(key='mixing_energy')
    
    opt = BayesianRidge(fit_intercept=False, compute_score=True)
    opt.fit(X, y)
    
    # 予測値
    y_pred = opt.predict(X)
    
    # Cross Validation
    cv_folds = min(5, len(y))
    scores = cross_val_score(opt, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
    cv_rmse = -np.mean(scores)
    train_rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    # ECI情報の抽出
    ecis = opt.coef_
    eci_data = []
    
    # Zerolet
    eci_data.append({
        'orbit_id': 0,
        'order': 0,
        'radius': 0.0,
        'multiplicity': 1,
        'eci_eV': ecis[0]
    })
    
    # その他のorbit
    for i, orbit in enumerate(cs.orbit_list):
        eci_data.append({
            'orbit_id': i + 1,
            'order': orbit.order,
            'radius': orbit.radius,
            'multiplicity': len(orbit),
            'eci_eV': ecis[i + 1]
        })
    
    # レポートデータを保存
    REPORT_DATA['n_structures'] = N_STRUCTURES
    REPORT_DATA['lattice_const'] = LATTICE_CONST
    REPORT_DATA['cutoffs'] = CUTOFFS
    REPORT_DATA['cv_rmse'] = cv_rmse
    REPORT_DATA['train_rmse'] = train_rmse
    REPORT_DATA['n_parameters'] = len(ecis)
    REPORT_DATA['concentrations'] = np.array(concentrations_list)
    REPORT_DATA['mixing_energies'] = np.array(mixing_energies_list)
    REPORT_DATA['y_actual'] = y
    REPORT_DATA['y_pred'] = y_pred
    REPORT_DATA['eci_df'] = pd.DataFrame(eci_data)
    REPORT_DATA['X'] = X
    REPORT_DATA['cluster_space_info'] = str(cs)
    
    print(f"計算完了: CV RMSE = {cv_rmse:.5f} eV/atom")
    
    return REPORT_DATA


def plot_eci_vs_radius(eci_df, save_path):
    """
    ECI値 vs クラスター半径の散布図
    orderで色分け、multiplicityでサイズを変更
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # order別の色
    colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd'}
    order_names = {0: 'Zerolet', 1: 'Singlet', 2: 'Pair', 3: 'Triplet', 4: 'Quadruplet'}
    
    for order in sorted(eci_df['order'].unique()):
        subset = eci_df[eci_df['order'] == order]
        sizes = subset['multiplicity'] * 30 + 50
        ax.scatter(
            subset['radius'], 
            subset['eci_eV'] * 1000,  # meVに変換
            s=sizes,
            c=colors.get(order, '#333333'),
            label=f'{order_names.get(order, f"Order {order}")} (n={len(subset)})',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Cluster Radius (Angstrom)', fontsize=12)
    ax.set_ylabel('ECI (meV)', fontsize=12)
    ax.set_title('Effective Cluster Interactions vs Cluster Radius', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def plot_eci_bar_chart(eci_df, save_path):
    """
    |ECI|の大きい上位クラスターの横棒グラフ
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # |ECI|でソート
    df_sorted = eci_df.copy()
    df_sorted['abs_eci'] = np.abs(df_sorted['eci_eV'])
    df_sorted = df_sorted.sort_values('abs_eci', ascending=True)
    
    # 色分け（正/負）
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in df_sorted['eci_eV']]
    
    # ラベル作成
    labels = [f"Order {row['order']}, r={row['radius']:.2f}" for _, row in df_sorted.iterrows()]
    
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos, df_sorted['eci_eV'] * 1000, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('ECI (meV)', fontsize=12)
    ax.set_title('Effective Cluster Interactions (All Clusters)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def plot_mixing_energy_vs_composition(concentrations, mixing_energies, save_path):
    """
    混合エネルギー vs 組成の散布図
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散布図
    ax.scatter(concentrations, mixing_energies * 1000, 
               c='#1f77b4', alpha=0.6, s=60, edgecolors='black', linewidths=0.5,
               label='DFT Data (Dummy)')
    
    # 理論曲線（正規溶体モデル）
    c_theory = np.linspace(0, 1, 100)
    omega = 0.15
    e_theory = omega * c_theory * (1 - c_theory)
    ax.plot(c_theory, e_theory * 1000, 'r-', linewidth=2, 
            label=f'Regular Solution Model (Omega={omega} eV)')
    
    ax.set_xlabel('V Concentration (x in Fe$_{1-x}$V$_x$)', fontsize=12)
    ax.set_ylabel('Mixing Energy (meV/atom)', fontsize=12)
    ax.set_title('Mixing Energy vs Composition', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def plot_parity(y_actual, y_pred, train_rmse, save_path):
    """
    予測値 vs 実測値のパリティプロット
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # meVに変換
    y_actual_mev = y_actual * 1000
    y_pred_mev = y_pred * 1000
    
    # 散布図
    ax.scatter(y_actual_mev, y_pred_mev, 
               c='#1f77b4', alpha=0.6, s=60, edgecolors='black', linewidths=0.5)
    
    # y=x線
    min_val = min(y_actual_mev.min(), y_pred_mev.min())
    max_val = max(y_actual_mev.max(), y_pred_mev.max())
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', linewidth=2, label='y = x')
    
    ax.set_xlabel('Actual Mixing Energy (meV/atom)', fontsize=12)
    ax.set_ylabel('Predicted Mixing Energy (meV/atom)', fontsize=12)
    ax.set_title(f'Parity Plot (RMSE = {train_rmse*1000:.2f} meV/atom)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def plot_residuals(y_actual, y_pred, concentrations, save_path):
    """
    残差プロット（残差 vs 濃度）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = (y_actual - y_pred) * 1000  # meV
    
    # 残差 vs 予測値
    ax1 = axes[0]
    ax1.scatter(y_pred * 1000, residuals, 
                c='#1f77b4', alpha=0.6, s=60, edgecolors='black', linewidths=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Predicted Mixing Energy (meV/atom)', fontsize=12)
    ax1.set_ylabel('Residual (meV/atom)', fontsize=12)
    ax1.set_title('Residuals vs Predicted Values', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 残差 vs 濃度
    ax2 = axes[1]
    ax2.scatter(concentrations, residuals, 
                c='#2ca02c', alpha=0.6, s=60, edgecolors='black', linewidths=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('V Concentration', fontsize=12)
    ax2.set_ylabel('Residual (meV/atom)', fontsize=12)
    ax2.set_title('Residuals vs Composition', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def plot_eci_distribution(eci_df, save_path):
    """
    ECI分布のヒストグラムとorder別箱ひげ図
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヒストグラム
    ax1 = axes[0]
    eci_values = eci_df['eci_eV'].values * 1000  # meV
    ax1.hist(eci_values, bins=15, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('ECI (meV)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('ECI Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Order別箱ひげ図
    ax2 = axes[1]
    order_names = {0: 'Zerolet', 1: 'Singlet', 2: 'Pair', 3: 'Triplet', 4: 'Quadruplet'}
    
    orders = sorted(eci_df['order'].unique())
    data_by_order = [eci_df[eci_df['order'] == o]['eci_eV'].values * 1000 for o in orders]
    labels = [order_names.get(o, f'Order {o}') for o in orders]
    
    bp = ax2.boxplot(data_by_order, labels=labels, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors[:len(orders)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Cluster Order', fontsize=12)
    ax2.set_ylabel('ECI (meV)', fontsize=12)
    ax2.set_title('ECI Distribution by Cluster Order', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def plot_cluster_schematic(save_path):
    """
    BCCクラスターの模式図
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # BCC格子の原子位置（2x2x2 supercell）
    a = 1.0  # 規格化
    
    # 角の原子
    corner_positions = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                corner_positions.append([i*a, j*a, k*a])
    
    # 体心の原子
    body_center_positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                body_center_positions.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    # 3D投影を2Dに変換（等角投影）
    def project_3d_to_2d(x, y, z):
        px = x - y * 0.5
        py = z + (x + y) * 0.3
        return px, py
    
    # 角の原子をプロット
    for pos in corner_positions:
        px, py = project_3d_to_2d(*pos)
        ax.scatter(px, py, s=200, c='#1f77b4', edgecolors='black', linewidths=1, zorder=5)
    
    # 体心の原子をプロット
    for pos in body_center_positions:
        px, py = project_3d_to_2d(*pos)
        ax.scatter(px, py, s=200, c='#ff7f0e', edgecolors='black', linewidths=1, zorder=5)
    
    # 最近接ペア（体心-角）を線で結ぶ
    center = body_center_positions[0]
    nn_corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    
    for corner in nn_corners:
        cx, cy = project_3d_to_2d(*center)
        px, py = project_3d_to_2d(*corner)
        ax.plot([cx, px], [cy, py], 'g-', linewidth=1.5, alpha=0.5, zorder=1)
    
    # 次近接ペア（角-角）の例
    ax.plot(*zip(project_3d_to_2d(0, 0, 0), project_3d_to_2d(1, 0, 0)), 
            'r--', linewidth=2, alpha=0.7, zorder=2)
    
    # 凡例
    ax.scatter([], [], s=200, c='#1f77b4', edgecolors='black', label='Corner atoms')
    ax.scatter([], [], s=200, c='#ff7f0e', edgecolors='black', label='Body-center atoms')
    ax.plot([], [], 'g-', linewidth=1.5, label='1st NN pair (r ~ a*sqrt(3)/2)')
    ax.plot([], [], 'r--', linewidth=2, label='2nd NN pair (r = a)')
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.3, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('BCC Lattice Structure and Cluster Types', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def generate_markdown_report(report_data, image_paths):
    """
    Markdownレポートを生成
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ECI表の作成
    eci_df = report_data['eci_df']
    eci_table = "| Orbit ID | Order | Radius (A) | Multiplicity | ECI (meV) |\n"
    eci_table += "|----------|-------|------------|--------------|----------|\n"
    for _, row in eci_df.iterrows():
        eci_table += f"| {int(row['orbit_id'])} | {int(row['order'])} | {row['radius']:.4f} | {int(row['multiplicity'])} | {row['eci_eV']*1000:.3f} |\n"
    
    report = f"""# icet CVM計算 詳細レポート

**生成日時**: {timestamp}

## 1. 概要

本レポートは、icetライブラリを使用したクラスター展開（Cluster Expansion; CE）計算の結果を報告します。
Fe-V BCC合金系を対象とし、ダミーDFTデータを用いてECI（Effective Cluster Interactions）を抽出しました。

### 1.1 計算条件

| パラメータ | 値 |
|-----------|-----|
| 対象系 | Fe-V BCC合金 |
| 格子定数 | {report_data['lattice_const']:.2f} A |
| 構造数 | {report_data['n_structures']} |
| ペアカットオフ | {report_data['cutoffs'][0]:.1f} A |
| 3体カットオフ | {report_data['cutoffs'][1]:.1f} A |
| 4体カットオフ | {report_data['cutoffs'][2]:.1f} A |
| パラメータ数（ECI数） | {report_data['n_parameters']} |
| 回帰モデル | BayesianRidge |

### 1.2 精度指標

| 指標 | 値 |
|------|-----|
| Training RMSE | {report_data['train_rmse']*1000:.3f} meV/atom |
| CV RMSE (5-fold) | {report_data['cv_rmse']*1000:.3f} meV/atom |

## 2. クラスター空間の構造

### 2.1 BCC格子とクラスタータイプ

![BCC Lattice Structure](assets/icet_cvm_report/cluster_schematic.png)

BCC（体心立方）格子では、以下のクラスタータイプが定義されます：

- **Zerolet (Order 0)**: 空クラスター（定数項）
- **Singlet (Order 1)**: 点クラスター（濃度依存項）
- **Pair (Order 2)**: ペアクラスター（2体相互作用）
  - 最近接ペア: r ~ a*sqrt(3)/2 ~ 2.49 A
  - 次近接ペア: r = a ~ 2.87 A
- **Triplet (Order 3)**: 3体クラスター
- **Quadruplet (Order 4)**: 4体クラスター

### 2.2 ClusterSpace情報

```
{report_data['cluster_space_info']}
```

## 3. 入力データの分析

### 3.1 混合エネルギー vs 組成

![Mixing Energy vs Composition](assets/icet_cvm_report/mixing_energy_vs_composition.png)

ダミーデータは正規溶体モデル（Regular Solution Model）に基づいて生成されました：

```
E_mix = Omega * c_Fe * c_V + noise
```

ここで Omega = 0.15 eV（正の値 = 相分離傾向）です。

**注意**: このダミーデータは組成のみに依存し、原子配置（短距離秩序）の情報を含みません。
実際のDFTデータでは、同じ組成でも配置によってエネルギーが異なります。

## 4. フィッティング結果

### 4.1 パリティプロット

![Parity Plot](assets/icet_cvm_report/parity_plot.png)

予測値と実測値の相関を示します。理想的なフィッティングでは、全ての点がy=x線上に乗ります。

### 4.2 残差分析

![Residuals](assets/icet_cvm_report/residuals.png)

残差プロットは、モデルの系統的な誤差を検出するのに有用です：

- **左図**: 残差 vs 予測値 - 予測値に依存した系統誤差がないか確認
- **右図**: 残差 vs 濃度 - 特定の濃度領域で誤差が大きくないか確認

## 5. ECI（有効クラスター相互作用）の分析

### 5.1 ECI一覧表

{eci_table}

### 5.2 ECI vs クラスター半径

![ECI vs Radius](assets/icet_cvm_report/eci_vs_radius.png)

各クラスターのECI値を半径に対してプロットしています。
点のサイズはmultiplicity（多重度）を表し、色はクラスターの次数（order）を表します。

### 5.3 ECI棒グラフ

![ECI Bar Chart](assets/icet_cvm_report/eci_bar_chart.png)

全クラスターのECI値を棒グラフで表示しています。
緑色は正のECI、赤色は負のECIを示します。

### 5.4 ECI分布

![ECI Distribution](assets/icet_cvm_report/eci_distribution.png)

- **左図**: 全ECIのヒストグラム
- **右図**: クラスター次数別の箱ひげ図

## 6. CVMへの投入に関する注意事項

### 6.1 基底関数の定義

icetはデフォルトで直交基底を使用します。CVMソルバーがIsing型基底（σ = ±1）を
期待する場合は、以下の変換が必要になる場合があります：

```
J_Ising = J_icet * (変換係数)
```

### 6.2 multiplicityの扱い

CVMソルバーの入力形式によって、ECIにmultiplicityを掛ける/割る変換が必要な場合があります：

- 「クラスター1個あたり」の形式: そのまま使用
- 「格子点あたり」の形式: multiplicityで割る

### 6.3 ダミーデータの限界

**重要**: 本レポートのECIはダミーデータ（正規溶体モデル）から得られたものであり、
実際のFe-V合金の相互作用を反映していません。

実際のCVM計算には、DFT計算から得られた配置依存のエネルギーデータが必要です。

## 7. 再現手順

```bash
# 1. 依存パッケージのインストール
pip install icet matplotlib pandas numpy scikit-learn

# 2. 計算の実行
python icet_cvm_calculation.py

# 3. レポートの生成
python generate_icet_cvm_report.py
```

## 8. 出力ファイル

| ファイル | 説明 |
|---------|------|
| fe_v_data.db | ASEデータベース（構造と混合エネルギー） |
| fe_v_eci_for_cvm.csv | ECI値（CSV形式） |
| docs/icet_cvm_calculation_report_ja.md | 本レポート |
| docs/assets/icet_cvm_report/*.png | 可視化図 |

## 9. 参考文献

1. icet公式ドキュメント: https://icet.materialsmodeling.org/
2. Sanchez, J.M., Ducastelle, F., Gratias, D. (1984). Physica A, 128, 334-350.
3. de Fontaine, D. (1994). Solid State Physics, 47, 33-176.

---

*本レポートはgenerate_icet_cvm_report.pyによって自動生成されました。*
"""
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"レポート保存: {REPORT_FILE}")


def main():
    """メイン処理"""
    print("="*60)
    print("icet CVM計算レポート生成")
    print("="*60)
    
    # ディレクトリ作成
    setup_directories()
    
    # 計算実行
    report_data = run_calculation()
    
    # 可視化
    print("\n可視化を生成中...")
    
    image_paths = {}
    
    # 1. ECI vs Radius
    path = os.path.join(ASSETS_DIR, "eci_vs_radius.png")
    plot_eci_vs_radius(report_data['eci_df'], path)
    image_paths['eci_vs_radius'] = path
    
    # 2. ECI Bar Chart
    path = os.path.join(ASSETS_DIR, "eci_bar_chart.png")
    plot_eci_bar_chart(report_data['eci_df'], path)
    image_paths['eci_bar_chart'] = path
    
    # 3. Mixing Energy vs Composition
    path = os.path.join(ASSETS_DIR, "mixing_energy_vs_composition.png")
    plot_mixing_energy_vs_composition(
        report_data['concentrations'], 
        report_data['mixing_energies'], 
        path
    )
    image_paths['mixing_energy_vs_composition'] = path
    
    # 4. Parity Plot
    path = os.path.join(ASSETS_DIR, "parity_plot.png")
    plot_parity(
        report_data['y_actual'], 
        report_data['y_pred'], 
        report_data['train_rmse'],
        path
    )
    image_paths['parity_plot'] = path
    
    # 5. Residuals
    path = os.path.join(ASSETS_DIR, "residuals.png")
    plot_residuals(
        report_data['y_actual'], 
        report_data['y_pred'], 
        report_data['concentrations'],
        path
    )
    image_paths['residuals'] = path
    
    # 6. ECI Distribution
    path = os.path.join(ASSETS_DIR, "eci_distribution.png")
    plot_eci_distribution(report_data['eci_df'], path)
    image_paths['eci_distribution'] = path
    
    # 7. Cluster Schematic
    path = os.path.join(ASSETS_DIR, "cluster_schematic.png")
    plot_cluster_schematic(path)
    image_paths['cluster_schematic'] = path
    
    # レポート生成
    print("\nMarkdownレポートを生成中...")
    generate_markdown_report(report_data, image_paths)
    
    print("\n" + "="*60)
    print("レポート生成完了")
    print("="*60)
    print(f"レポートファイル: {REPORT_FILE}")
    print(f"画像ディレクトリ: {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
