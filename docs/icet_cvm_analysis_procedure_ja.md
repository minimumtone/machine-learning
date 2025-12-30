# icetを使ったCVM計算用ECI抽出 - 解析手順書

## 概要

本ドキュメントは、icetライブラリを使用してクラスター展開（Cluster Expansion; CE）を行い、CVM（Cluster Variation Method; クラスター変分法）計算に必要なECI（Effective Cluster Interactions; 有効クラスター相互作用）を抽出するための解析手順を詳細に説明します。

## 1. 理論的背景

### 1.1 CVM（クラスター変分法）とは

CVMは、合金の自由エネルギー G = E - TS を、局所クラスター（点・ペア・三角形・四面体など）の出現確率（相関）で表現し、エントロピー S をそのクラスター近似レベルで評価して、自由エネルギーを最小化することで平衡状態（短距離秩序や相分離傾向、相図）を求める方法です。

CVMが必要とする「エネルギー側の入力」は、配置（どの格子点にどの原子が入るか）に対するエネルギー関数であり、これを少数のパラメータで表現したものがクラスター展開（CE）です。

### 1.2 クラスター展開（CE）とは

CEでは、ある配置のエネルギー（ここでは混合エネルギー）を、点・ペア・多体クラスターの相関関数の線形結合として近似します：

```
E(σ) = Σ_α J_α × Φ_α(σ)
```

ここで：
- σ: 原子配置
- J_α: ECI（有効クラスター相互作用）
- Φ_α(σ): クラスター相関関数

この線形結合の係数がECIで、CVMはこのECIを使ってエネルギーEを計算し、エントロピーSは近似レベルに応じたCVMの式で与える、という役割分担になります。

## 2. 解析ワークフロー

```
[DFT計算] → [混合エネルギー計算] → [ASE DB作成] → [ClusterSpace構築]
    ↓                                                      ↓
多数の原子配置                                         母格子・カットオフ定義
(様々な濃度・秩序度)                                          ↓
                                                  [StructureContainer]
                                                         ↓
                                                  [回帰フィッティング]
                                                         ↓
                                                  [ECI抽出・CSV出力]
                                                         ↓
                                                  [CVMソルバーへ投入]
```

## 3. 詳細手順

### 3.1 ステップ1: DFTデータの準備

#### 3.1.1 必要なDFT計算

多数の原子配置（対象合金系の様々な濃度・秩序度・格子サイズ）についてDFT計算を実行します。

推奨される構造数：
- 最低20〜30構造（Cross Validationのため）
- 理想的には50〜100構造以上
- 濃度は0%〜100%の範囲を均等にカバー

#### 3.1.2 参照エネルギーの計算

純物質の参照エネルギー（同一のDFT条件で計算した、原子1個あたりのエネルギー）を準備します：

```python
E_FE = -8.305  # 純Fe (BCC) の参照エネルギー [eV/atom]
E_V  = -9.123  # 純V (BCC) の参照エネルギー [eV/atom]
```

**重要**: 参照エネルギーは、混合エネルギー計算に使用する全ての構造と同一のDFT条件（磁性状態、体積、k点、カットオフなど）で計算する必要があります。

#### 3.1.3 混合エネルギーの計算

各構造の混合（形成）エネルギーを以下の式で計算します：

```
E_mix = (E_total - N_Fe × E_Fe - N_V × E_V) / N_total
```

ここで：
- E_total: DFT計算から得られた全エネルギー
- E_Fe, E_V: 純Fe、純Vの参照エネルギー
- N_Fe, N_V, N_total: 構造中の各原子数

### 3.2 ステップ2: ASEデータベースの作成

ASE（Atomic Simulation Environment）のデータベース機能を使用して、構造と混合エネルギーを一元管理します：

```python
from ase.db import connect

db = connect("fe_v_data.db")
db.write(atoms, key_value_pairs={
    'mixing_energy': mixing_energy,
    'concentration_v': n_V / n_total
})
```

### 3.3 ステップ3: クラスター空間の構築

#### 3.3.1 母格子の定義

```python
from ase.build import bulk
from icet import ClusterSpace

prim = bulk('Fe', 'bcc', a=2.87)  # 格子定数 [Å]
```

#### 3.3.2 ClusterSpaceの作成

```python
cs = ClusterSpace(
    structure=prim,
    cutoffs=[6.0, 4.0, 4.0],  # [ペア, 3体, 4体] のカットオフ [Å]
    chemical_symbols=['Fe', 'V']
)
```

**カットオフパラメータの選択指針**:

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| ペアカットオフ | 2体相互作用の最大距離 | 5.0〜8.0 Å |
| 3体カットオフ | 3体相互作用の最大距離 | 3.0〜5.0 Å |
| 4体カットオフ | 4体相互作用の最大距離 | 3.0〜5.0 Å |

カットオフを大きくすると表現力は上がりますが、ECIの数が増え、必要なDFT構造数も増え、過学習のリスクも高まります。

### 3.4 ステップ4: 学習データの準備

```python
from icet import StructureContainer

sc = StructureContainer(cluster_space=cs)

for row in db.select():
    sc.add_structure(
        structure=row.toatoms(),
        user_tag=str(row.id),
        properties={'mixing_energy': row.mixing_energy}
    )

X, y = sc.get_fit_data(key='mixing_energy')
```

ここで：
- X: 特徴量行列（各構造のクラスター相関）
- y: ターゲットベクトル（混合エネルギー）

### 3.5 ステップ5: 回帰フィッティング

```python
from sklearn.linear_model import BayesianRidge

opt = BayesianRidge(fit_intercept=False, compute_score=True)
opt.fit(X, y)
```

**BayesianRidgeを使用する理由**:
- L1正則化（Lasso）のように極端にスパース化しにくい
- 安定して係数が出やすい
- CVMでは「スパースすぎない」ECIが扱いやすい場合が多い

#### 精度の確認

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(opt, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse = -np.mean(scores)
print(f"CV RMSE: {rmse:.5f} eV/atom")
```

目安：
- RMSE < 5 meV/atom: 良好
- RMSE < 10 meV/atom: 許容範囲
- RMSE > 20 meV/atom: カットオフやデータの見直しが必要

### 3.6 ステップ6: ECI出力

```python
ecis = opt.coef_

for i, orbit in enumerate(cs.orbit_list):
    order = orbit.order
    radius = orbit.radius
    multiplicity = len(orbit)
    eci = ecis[i + 1]  # ecis[0]はzerolet
```

## 4. 出力ファイルの解釈

### 4.1 CSVファイルの構造

| カラム | 説明 |
|--------|------|
| orbit_id | クラスターの識別番号 |
| order | クラスターの次数（0=空, 1=点, 2=ペア, 3=3体, 4=4体） |
| radius | クラスターの半径 [Å] |
| multiplicity | 単位格子あたりの多重度 |
| eci_eV | ECI値 [eV] |

### 4.2 orderの意味

| order | 名称 | 物理的意味 |
|-------|------|-----------|
| 0 | 空クラスター（zerolet） | エネルギー基準（定数項） |
| 1 | 点クラスター（singlet） | 濃度依存項 |
| 2 | ペアクラスター | 2体相互作用 |
| 3 | 3体クラスター（triplet） | 3体相互作用 |
| 4 | 4体クラスター（quadruplet） | 4体相互作用 |

### 4.3 radiusの意味

- ペアの場合：原子間距離の半分
- 多体の場合：重心からの最大距離
- BCC構造の場合の典型的な値：
  - 最近接ペア：約1.24 Å（原子間距離 ≈ 2.49 Å）
  - 次近接ペア：約1.44 Å（原子間距離 ≈ 2.87 Å）

## 5. CVMへの投入時の注意点

### 5.1 基底関数の定義

二元系CEでは、占有変数を σ = ±1 とするIsing型表現や、直交基底での表現などがあり、同じ「ペアECI」でも解釈が異なる場合があります。

icetはデフォルトで直交基底を使用しますが、CVMソルバーがIsing型を期待する場合は変換が必要です。

### 5.2 multiplicityの扱い

CVMソルバーが期待する形式によって、ECIにmultiplicityを掛ける/割る変換が必要な場合があります：

- 「クラスター1個あたり」の形式：そのまま使用
- 「格子点あたり」の形式：multiplicityで割る

### 5.3 検証手順

1. まずは最近接ペア程度の小さい項だけを使用
2. 予想される傾向（秩序化/相分離）が直感と合うか確認
3. 徐々にクラスターを追加して精度を向上

## 6. トラブルシューティング

### 6.1 RMSEが大きい場合

- DFT構造数を増やす
- カットオフを調整（大きすぎると過学習、小さすぎると表現力不足）
- 参照エネルギーの計算条件を確認

### 6.2 ECIが不安定な場合

- 正則化パラメータを調整
- 異なる回帰モデル（Ridge, Lasso, ElasticNet）を試す
- 外れ値となる構造を確認・除外

### 6.3 CVMで予想と異なる結果が出る場合

- 基底関数の定義を確認
- multiplicityの扱いを確認
- 符号規約を確認（異種原子を好む/同種原子を好む）

## 7. 参考文献

1. icet公式ドキュメント: https://icet.materialsmodeling.org/
2. Sanchez, J.M., Ducastelle, F., Gratias, D. (1984). Physica A, 128, 334-350.
3. de Fontaine, D. (1994). Solid State Physics, 47, 33-176.

## 8. 実行例

```bash
# スクリプトの実行
python icet_cvm_calculation.py

# 出力ファイル
# - fe_v_data.db: ASEデータベース
# - fe_v_eci_for_cvm.csv: ECI値
```
