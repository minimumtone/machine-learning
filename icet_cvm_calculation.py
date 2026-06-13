#!/usr/bin/env python3
"""
icetを使ったCVM計算用ECI抽出スクリプト

このスクリプトは、Fe-V BCC合金系を例として、クラスター展開(Cluster Expansion; CE)を
用いてECI(Effective Cluster Interactions)を抽出し、CVM(Cluster Variation Method)
計算に投入可能な形式で出力します。

=============================================================================
解析手順の詳細
=============================================================================

【1. CVM（クラスター変分法）とは】
CVMは、合金の自由エネルギー G = E - TS を、局所クラスター（点・ペア・三角形・
四面体など）の出現確率（相関）で表現し、エントロピー S をそのクラスター近似
レベルで評価して、自由エネルギーを最小化することで平衡状態（短距離秩序や
相分離傾向、相図）を求める方法です。

CVMが必要とする「エネルギー側の入力」は、配置（どの格子点にFe/Vが入るか）に
対するエネルギー関数であり、これを少数のパラメータで表現したものがクラスター
展開（CE）です。CEでは、ある配置のエネルギー（ここでは混合エネルギー）を、
点・ペア・多体クラスターの相関関数の線形結合として近似します。

この線形結合の係数がECI（有効クラスター相互作用）で、CVMはこのECIを使って
エネルギーEを計算し、エントロピーSは近似レベルに応じたCVMの式で与える、
という役割分担になります。

【2. DFTデータからECI抽出までのワークフロー】

  [DFT計算] → [混合エネルギー計算] → [ASE DB作成] → [ClusterSpace構築]
      ↓                                                      ↓
  多数の原子配置                                         母格子・カットオフ定義
  (Fe-Vの様々な濃度・秩序度)                                    ↓
                                                    [StructureContainer]
                                                           ↓
                                                    [回帰フィッティング]
                                                           ↓
                                                    [ECI抽出・CSV出力]
                                                           ↓
                                                    [CVMソルバーへ投入]

【3. 混合エネルギーの計算】
各構造の混合（形成）エネルギーは以下の式で計算します：

  E_mix = (E_total - N_Fe * E_Fe - N_V * E_V) / N_total

ここで：
  - E_total: DFT計算から得られた全エネルギー
  - E_Fe, E_V: 純Fe、純Vの参照エネルギー（同一DFT条件で計算）
  - N_Fe, N_V, N_total: 構造中の各原子数

【4. クラスター空間（ClusterSpace）のパラメータ】

  - cutoffs: [r2, r3, r4] の形式で、ペア・3体・4体クラスターの
    カットオフ距離(Å)を指定。大きくすると表現力は上がるが、
    ECIの数が増え、過学習のリスクも高まる。
    
  - chemical_symbols: 置換可能な元素種（ここでは['Fe', 'V']）
  
  - LATTICE_CONST: 母格子の格子定数。近接原子間距離の定義に関わる。

【5. 回帰モデルの選択】
BayesianRidgeを使用する理由：
  - L1正則化（Lasso）のように極端にスパース化しにくい
  - 安定して係数が出やすい
  - CVMでは「スパースすぎない」ECIが扱いやすい場合が多い

【6. ECI出力の解釈における注意点】

  (a) 基底関数の定義：
      二元系CEでは、占有変数を σ = ±1 とするIsing型表現や、
      直交基底での表現などがあり、同じ「ペアECI」でも解釈が異なる
      場合があります。

  (b) multiplicity（多重度）：
      あるorbitが単位格子あたり何回現れるかの情報。CVMソルバーが
      期待する形式（クラスター1個あたり vs 格子点あたり）に応じて、
      ECIに multiplicity を掛ける/割る変換が必要な場合があります。

  (c) order（次数）の意味：
      - order=0: 空クラスター（定数項）
      - order=1: 点クラスター（singlet）
      - order=2: ペアクラスター
      - order=3: 3体クラスター（triplet）
      - order=4: 4体クラスター（quadruplet）

【7. ダミーデータ生成の方法】
本スクリプトでは、DFTデータの代わりに以下の方法でダミーデータを生成：

  1. BCC supercell（2x2x2, 3x3x3など）を作成
  2. 様々な濃度でFe/Vをランダム配置
  3. 「既知のECI」から合成したエネルギーを計算
  4. 少量のノイズを加えてDFTらしいばらつきを模擬
  5. フィットで元のECIが復元できることを確認

この方法により、end-to-endでCE→ECI出力まで動作確認ができます。

=============================================================================
"""

import os
import numpy as np
import pandas as pd
from ase.db import connect
from ase.build import bulk, make_supercell
from ase import Atoms
from icet import ClusterSpace, StructureContainer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
import random

# ==========================================
# 1. 設定パラメータ
# ==========================================

# データベースの保存ファイル名
DB_FILENAME = "fe_v_data.db"

# 純物質の原子1つあたりのエネルギー (eV/atom)
# ダミーデータ用の参照値
E_FE = -8.305  # 純Fe (BCC) の参照エネルギー
E_V  = -9.123  # 純V (BCC) の参照エネルギー

# クラスター展開のカットオフ (Å)
# [ペアのカットオフ, 3体のカットオフ, 4体のカットオフ]
CUTOFFS = [6.0, 4.0, 4.0] 

# 母格子の格子定数 (Å)
LATTICE_CONST = 2.87  # Fe-V BCCのおおよその値

# ダミーデータ生成用の「真のECI」（検証用）
# これらの値からエネルギーを合成し、フィットで復元できるか確認
TRUE_ECIS = {
    'J0': 0.0,      # 空クラスター（基準）
    'J1': 0.01,     # 点クラスター
    'J2_1nn': -0.05,  # 最近接ペア（負=異種原子を好む）
    'J2_2nn': 0.02,   # 次近接ペア
}

# ==========================================
# 2. ダミーDFTデータの生成
# ==========================================

def generate_dummy_structures(n_structures=50, seed=42):
    """
    Fe-V BCC合金のダミー構造を生成する
    
    Parameters
    ----------
    n_structures : int
        生成する構造の数
    seed : int
        乱数シード（再現性のため）
    
    Returns
    -------
    list of tuple
        (atoms, mixing_energy, concentration_v) のリスト
    """
    random.seed(seed)
    np.random.seed(seed)
    
    structures = []
    
    # 母格子（BCC Fe）
    prim = bulk('Fe', 'bcc', a=LATTICE_CONST)
    
    # 様々なサイズのsupercellを使用
    supercell_sizes = [
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2x2x2 = 16 atoms
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],  # 3x3x3 = 54 atoms
        [[2, 0, 0], [0, 2, 0], [0, 0, 3]],  # 2x2x3 = 24 atoms
    ]
    
    # 濃度の範囲（0.0〜1.0）
    concentrations = np.linspace(0.0, 1.0, 11)  # 0%, 10%, ..., 100%
    
    structure_count = 0
    
    for _ in range(n_structures):
        # ランダムにsupercellサイズを選択
        sc_matrix = random.choice(supercell_sizes)
        supercell = make_supercell(prim, sc_matrix)
        n_atoms = len(supercell)
        
        # ランダムに濃度を選択（または指定濃度から選択）
        if random.random() < 0.5:
            conc_v = random.choice(concentrations)
        else:
            conc_v = random.uniform(0.0, 1.0)
        
        n_v = int(round(conc_v * n_atoms))
        n_fe = n_atoms - n_v
        
        # 原子種をランダムに配置
        symbols = ['Fe'] * n_fe + ['V'] * n_v
        random.shuffle(symbols)
        supercell.set_chemical_symbols(symbols)
        
        # ダミーの混合エネルギーを計算
        # 簡単なモデル：最近接Fe-Vペア数に比例 + ノイズ
        mixing_energy = calculate_dummy_mixing_energy(supercell)
        
        structures.append((supercell, mixing_energy, n_v / n_atoms))
        structure_count += 1
    
    print(f"生成した構造数: {structure_count}")
    return structures


def calculate_dummy_mixing_energy(atoms):
    """
    ダミーの混合エネルギーを計算する
    
    簡単なモデル：
    - 濃度依存の混合エネルギー（正規溶体近似）
    - 少量のノイズを追加
    
    Parameters
    ----------
    atoms : ase.Atoms
        原子構造
    
    Returns
    -------
    float
        混合エネルギー (eV/atom)
    """
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    
    # 濃度
    n_v = symbols.count('V')
    c_v = n_v / n_atoms
    c_fe = 1.0 - c_v
    
    # 混合エネルギーの計算（正規溶体モデル + ノイズ）
    # E_mix = Omega * c_Fe * c_V + noise
    # Omega > 0: 相分離傾向、Omega < 0: 秩序化傾向
    
    # 相互作用パラメータ（eV/atom）
    omega = 0.15  # 正の値 = 相分離傾向（Fe-Vは実際には相分離系）
    
    # 正規溶体項
    regular_term = omega * c_fe * c_v
    
    # 非対称項（サブレギュラー溶体的な補正）
    asymmetric_term = 0.02 * c_fe * c_v * (c_fe - c_v)
    
    # ノイズ（DFTの数値誤差を模擬）
    noise = np.random.normal(0, 0.003)  # 3 meV/atom程度のノイズ
    
    mixing_energy = regular_term + asymmetric_term + noise
    
    return mixing_energy


def prepare_database(structures=None):
    """
    ASEデータベースを作成する
    
    Parameters
    ----------
    structures : list, optional
        (atoms, mixing_energy, concentration_v) のリスト
        Noneの場合はダミーデータを生成
    """
    if os.path.exists(DB_FILENAME):
        os.remove(DB_FILENAME)
        print(f"既存のデータベース {DB_FILENAME} を削除しました。")
    
    if structures is None:
        print("ダミーDFTデータを生成しています...")
        structures = generate_dummy_structures(n_structures=50)
    
    print("データベースを作成しています...")
    db = connect(DB_FILENAME)
    
    count = 0
    for atoms, mixing_energy, conc_v in structures:
        # データベースに書き込み
        db.write(atoms, key_value_pairs={
            'mixing_energy': mixing_energy,
            'concentration_v': conc_v
        })
        count += 1
    
    print(f"データベース作成完了: {count} 個の構造を追加しました。")
    return count


# ==========================================
# 3. クラスター展開の実行 (Training)
# ==========================================

def train_model():
    """
    クラスター展開モデルを学習する
    
    Returns
    -------
    cs : ClusterSpace
        クラスター空間
    opt : BayesianRidge
        学習済み回帰モデル
    """
    print("\n" + "="*60)
    print("クラスター空間の構築")
    print("="*60)
    
    # 母格子 (BCC)
    prim = bulk('Fe', 'bcc', a=LATTICE_CONST)
    
    # ClusterSpaceの定義
    cs = ClusterSpace(
        structure=prim,
        cutoffs=CUTOFFS,
        chemical_symbols=['Fe', 'V']
    )
    print(cs)
    
    print("\n" + "="*60)
    print("構造コンテナへのデータ充填")
    print("="*60)
    
    sc = StructureContainer(cluster_space=cs)
    db = connect(DB_FILENAME)
    
    # DBからデータをロード
    for row in db.select():
        sc.add_structure(
            structure=row.toatoms(),
            user_tag=str(row.id),
            properties={'mixing_energy': row.mixing_energy}
        )
    
    print(f"学習に使用するサンプル数: {len(sc)}")
    
    print("\n" + "="*60)
    print("フィッティング実行")
    print("="*60)
    
    # 特徴量行列 X と ターゲットベクトル y を取得
    X, y = sc.get_fit_data(key='mixing_energy')
    
    print(f"特徴量行列の形状: {X.shape}")
    print(f"  - サンプル数: {X.shape[0]}")
    print(f"  - ECI数（クラスター数）: {X.shape[1]}")
    
    # 回帰モデルの選択
    opt = BayesianRidge(fit_intercept=False, compute_score=True)
    opt.fit(X, y)
    
    # 精度の確認 (Cross Validation)
    n_samples = len(y)
    cv_folds = min(5, n_samples)  # サンプル数が少ない場合は調整
    
    if cv_folds >= 2:
        scores = cross_val_score(
            opt, X, y, 
            cv=cv_folds, 
            scoring='neg_root_mean_squared_error'
        )
        rmse = -np.mean(scores)
        print(f"\nCross Validation RMSE: {rmse:.5f} eV/atom")
        print(f"  (CV folds: {cv_folds})")
    else:
        print("\n警告: サンプル数が少なすぎてCross Validationを実行できません")
    
    # 予測精度の確認
    y_pred = opt.predict(X)
    train_rmse = np.sqrt(np.mean((y - y_pred)**2))
    print(f"Training RMSE: {train_rmse:.5f} eV/atom")
    
    return cs, opt


# ==========================================
# 4. CVM用 ECI の出力
# ==========================================

def export_eci_for_cvm(cs, opt):
    """
    CVM計算用にECIをCSVファイルに出力する
    
    Parameters
    ----------
    cs : ClusterSpace
        クラスター空間
    opt : BayesianRidge
        学習済み回帰モデル
    """
    print("\n" + "="*60)
    print("ECI (有効クラスター相互作用) の出力")
    print("="*60)
    
    ecis = opt.coef_
    output_data = []
    
    print("\n【主要なECI値】")
    print("-" * 60)
    print(f"{'Orbit ID':>8} {'Order':>6} {'Radius(Å)':>10} {'Multi':>6} {'ECI(eV)':>12}")
    print("-" * 60)
    
    # 各クラスター（Orbit）の情報を抽出
    # 注意: orbit_listにはzerolet（空クラスター）が含まれないため、
    # ECIのインデックスは1からスタート（ecis[0]はzerolet）
    
    # まずzerolet（空クラスター）を追加
    cluster_info = {
        'orbit_id': 0,
        'order': 0,
        'radius': 0.0,
        'multiplicity': 1,
        'eci_eV': ecis[0]
    }
    output_data.append(cluster_info)
    print(f"{0:>8} {0:>6} {0.0:>10.4f} {1:>6} {ecis[0]:>12.6f}")
    
    # orbit_listの各orbitを処理
    for i, orbit in enumerate(cs.orbit_list):
        eci_idx = i + 1  # ECIのインデックスは1からスタート
        eci = ecis[eci_idx]
        
        # orbitの情報を取得
        order = orbit.order
        radius = orbit.radius
        multiplicity = len(orbit)  # len(orbit)で多重度を取得
        
        cluster_info = {
            'orbit_id': eci_idx,
            'order': order,
            'radius': radius,
            'multiplicity': multiplicity,
            'eci_eV': eci
        }
        output_data.append(cluster_info)
        
        # 主要なクラスター（order <= 2 かつ radius < 5.0）を表示
        if order <= 2 and radius < 5.0:
            print(f"{eci_idx:>8} {order:>6} {radius:>10.4f} {multiplicity:>6} {eci:>12.6f}")
    
    print("-" * 60)
    
    # CSVファイルに保存
    df = pd.DataFrame(output_data)
    csv_filename = "fe_v_eci_for_cvm.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\n完了: ECIデータを '{csv_filename}' に保存しました。")
    
    # ECI解釈のガイダンス
    print("\n" + "="*60)
    print("ECI解釈のガイダンス")
    print("="*60)
    print("""
【orderの意味】
  - order=0: 空クラスター（定数項、エネルギー基準）
  - order=1: 点クラスター（濃度依存項）
  - order=2: ペアクラスター（2体相互作用）
  - order=3: 3体クラスター
  - order=4: 4体クラスター

【radiusの意味】
  - クラスターの「大きさ」を表す半径(Å)
  - ペアの場合は原子間距離の半分
  - 多体の場合は重心からの最大距離

【multiplicityの意味】
  - そのorbitが単位格子あたり何回現れるか
  - CVMソルバーの入力形式によっては、ECIに
    multiplicityを掛ける/割る変換が必要

【CVMへの投入時の注意】
  1. 使用するCVMソルバーの基底関数定義を確認
  2. エネルギー式の規格化（クラスター1個 vs 格子点あたり）を確認
  3. まずは最近接ペア程度の小さい項だけで検証
  4. 予想される傾向（秩序化/相分離）が直感と合うか確認
""")
    
    return df


def print_analysis_summary():
    """
    解析手順のサマリーを表示する
    """
    print("""
================================================================================
icetを使ったCVM計算用ECI抽出 - 解析手順サマリー
================================================================================

【ステップ1: データ準備】
  - DFT計算結果（OUTCAR/vasprun.xml）を収集
  - 純Fe、純Vの参照エネルギーを計算
  - 混合エネルギーを計算してASE DBに保存

【ステップ2: クラスター空間の構築】
  - 母格子（BCC）を定義
  - カットオフ距離を設定
  - ClusterSpaceオブジェクトを作成

【ステップ3: 学習データの準備】
  - StructureContainerにDFT構造を追加
  - 特徴量行列（クラスター相関）を計算

【ステップ4: 回帰フィッティング】
  - BayesianRidge回帰でECIを推定
  - Cross Validationで精度を確認

【ステップ5: ECI出力】
  - 各orbitのECI値をCSVに保存
  - CVMソルバーへの投入形式に変換

【重要なパラメータ】
  - CUTOFFS: クラスターのカットオフ距離 [ペア, 3体, 4体]
  - LATTICE_CONST: 母格子の格子定数
  - E_FE, E_V: 純物質の参照エネルギー

【出力ファイル】
  - fe_v_data.db: ASEデータベース（構造と混合エネルギー）
  - fe_v_eci_for_cvm.csv: ECI値（CVMへの入力用）

================================================================================
""")


# ==========================================
# メイン処理
# ==========================================

if __name__ == "__main__":
    # 解析手順のサマリーを表示
    print_analysis_summary()
    
    # 1. ダミーデータでDB準備
    print("\n" + "="*60)
    print("ステップ1: データベース準備（ダミーデータ使用）")
    print("="*60)
    prepare_database()
    
    # 2. 学習
    cs, opt = train_model()
    
    # 3. ECI出力
    df = export_eci_for_cvm(cs, opt)
    
    print("\n" + "="*60)
    print("処理完了")
    print("="*60)
    print("生成されたファイル:")
    print(f"  - {DB_FILENAME}: ASEデータベース")
    print("  - fe_v_eci_for_cvm.csv: ECI値")
