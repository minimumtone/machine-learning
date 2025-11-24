# 結晶構造解析アプリケーション 詳細ドキュメント

## Crystal Structure Analysis Application - Comprehensive Documentation

**バージョン:** 1.0.0  
**作成日:** 2025年11月24日  
**開発:** Devin AI  
**ライセンス:** MIT License

---

## 目次

1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [数学的基礎](#数学的基礎)
4. [実装詳細](#実装詳細)
5. [使用方法](#使用方法)
6. [検証戦略](#検証戦略)
7. [参考文献](#参考文献)

---

## 概要

### プロジェクトの目的

本アプリケーションは、材料工学における結晶構造の「無秩序性（Disorder）」を、現代的なマテリアルズ・インフォマティクス（MI）の観点から可視化・計測するWebアプリケーションです。

特に、単に計算するだけでなく、**幾何学的な定義（直積・直和）**や**対称性（E(3)不変性）**といった数学的概念が、実際のプログラムコードにどう落とし込まれるかを、操作しながら学習できるツールを目指しています。

### ターゲットユーザー

- Python基礎はあるが、材料工学や幾何学的深層学習（Geometric Deep Learning）は初心者のエンジニア
- 結晶構造の無秩序性を定量的に評価したい研究者
- マテリアルズ・インフォマティクスに興味がある学生

### コアコンセプト

1. **Math-First:** 数学的定義をコードのクラス設計に反映させる
2. **Data Reduction:** 計算コストを抑え、本質（ロジック）の理解を優先する
3. **Visual Verification:** 数値だけでなく、GUI上で網羅的なパターンを目視検証する

---

## 理論的背景

### 結晶構造と無秩序性

結晶構造は、原子が規則的に配列した固体の構造です。しかし、実際の材料では、以下のような無秩序性が存在します：

1. **化学的無秩序（Chemical Disorder）:** 異なる原子種がランダムに配置
2. **位置的無秩序（Positional Disorder）:** 原子位置のゆらぎ
3. **配向的無秩序（Orientational Disorder）:** 分子の配向のランダム性

本アプリケーションでは、**化学的無秩序**に焦点を当て、Warren-Cowley Short Range Order (SRO) パラメータを用いて定量化します。

### Warren-Cowley SRO パラメータ

Warren-Cowley SROパラメータ α は、1950年にCowleyによって提案され、合金の短範囲秩序を定量化する指標として広く使用されています。

#### 定義

$$\alpha_n = 1 - \frac{P_n(B|A)}{c_B}$$

ここで：
- $\alpha_n$: 第n近接殻におけるSROパラメータ
- $P_n(B|A)$: A原子の第n近接位置にB原子が存在する条件付き確率
- $c_B$: B原子の全体濃度

#### 物理的意味

| α の値 | 物理的意味 | 原子配置の特徴 |
|--------|-----------|---------------|
| α ≈ 0 | ランダム構造 | 統計的に完全にランダムな配置 |
| α < 0 | 規則構造 | A-B交互配列（規則合金） |
| α > 0 | クラスター構造 | 同種原子が集まる（相分離傾向） |

#### 熱力学との関係

- **α < 0 (規則構造):** 負の混合エンタルピー → エンタルピー的に安定
- **α > 0 (クラスター構造):** 正の混合エンタルピー → 相分離傾向
- **α ≈ 0 (ランダム構造):** エントロピーが最大 → 高温で安定

---

## 数学的基礎

### 線形代数による結晶構造の記述

#### 1. 格子空間 (Lattice Space)

結晶格子は、3つの基本並進ベクトル $\mathbf{a}, \mathbf{b}, \mathbf{c}$ によって張られる空間として定義されます：

$$\mathcal{L} = \text{span}(\mathbf{a}, \mathbf{b}, \mathbf{c}) = \{n_1\mathbf{a} + n_2\mathbf{b} + n_3\mathbf{c} \mid n_1, n_2, n_3 \in \mathbb{Z}\}$$

**数学的意味:**
- **span（生成空間）:** ベクトルの線形結合で表される空間
- **離散性:** 整数係数のみ → 並進対称性

**プログラム実装:**
```python
def _get_basis_vectors(self) -> np.ndarray:
    a = self.lattice_constant
    return np.array([
        [a, 0, 0],  # a vector
        [0, a, 0],  # b vector
        [0, 0, a]   # c vector
    ])
```

#### 2. 原子の状態 (Atomic State)

1つの原子 $i$ の状態 $s_i$ は、位置座標 $x_i$ と化学種属性 $f_i$ の**直積（Tensor Product）**として表現されます：

$$s_i = x_i \otimes f_i \quad (x_i \in \mathbb{R}^3, f_i \in \{0, 1\})$$

**数学的意味:**
- **直積 ⊗:** 2つの空間の直積空間への写像
- **独立性:** 位置と化学種は独立した自由度

**プログラム実装:**
```python
# 位置と化学種を別々の配列として管理
self.positions = np.array([...])  # N × 3 array
self.species = np.array([...])    # N array (0 or 1)
```

#### 3. 全系の状態 (System State)

結晶全体の状態 $S$ は、全原子状態の**直和（Direct Sum）**として表現されます：

$$S = \bigoplus_{i=1}^{N} s_i = \bigoplus_{i=1}^{N} (x_i \otimes f_i)$$

**数学的意味:**
- **直和 ⊕:** 各成分が独立した状態空間の和
- **全体性:** 全原子の状態を包含

**プログラム実装:**
```python
def get_state_representation(self) -> Dict:
    return {
        'positions': self.positions,  # 全原子の位置
        'species': self.species,      # 全原子の化学種
        'n_atoms': len(self.positions)
    }
```

### E(3) 不変性と対称性

#### ユークリッド群 E(3)

E(3)は、3次元ユークリッド空間における**等長変換**（距離を保存する変換）の群です：

$$E(3) = \{(R, \mathbf{t}) \mid R \in SO(3), \mathbf{t} \in \mathbb{R}^3\}$$

ここで：
- $R$: 回転行列（$SO(3)$: 特殊直交群）
- $\mathbf{t}$: 並進ベクトル

#### E(3) 不変性の定義

物理量 $f$ が E(3) 不変であるとは：

$$f(R\{x_i\} + \mathbf{t}) = f(\{x_i\}) \quad \forall (R, \mathbf{t}) \in E(3)$$

#### Warren-Cowley SRO の E(3) 不変性

Warren-Cowley SRO パラメータ $\alpha$ は、**スカラー量**であり、E(3)不変です：

**証明の概略:**
1. $\alpha$ は原子間距離のみに依存
2. 距離は回転・並進に対して不変
3. したがって、$\alpha$ は E(3) 不変

**物理的意義:**
- 結晶の向きや位置に依存しない本質的な性質
- 実験測定値との対応が明確
- 機械学習における**幾何学的深層学習 (Geometric Deep Learning)** の基礎

#### 実装における E(3) 不変性の確認

```python
# 距離行列の計算（回転・並進不変）
distance_matrix = squareform(pdist(positions, metric='euclidean'))

# αの計算は距離行列のみに依存 → E(3)不変
alpha = self.calculate_alpha(shell=1)
```

---

## 実装詳細

### クラス設計

#### CrystalGeometry クラス

**責務:**
- 結晶構造の生成（SC, BCC, FCC）
- 原子種の割り当て
- 距離計算
- 状態表現の提供

**主要メソッド:**

1. `__init__(structure_type, size)`: 初期化
2. `_generate_positions()`: 原子位置の生成
3. `assign_species(concentration_B, random_seed)`: 化学種の割り当て
4. `calculate_neighbor_distances()`: 距離行列の計算
5. `get_coordination_number()`: 配位数の取得

**設計原則:**
- **単一責任:** 結晶構造の幾何学的側面のみを扱う
- **不変性:** 一度生成した位置は変更しない
- **再現性:** random_seedによる再現可能性

#### WarrenCowleySRO クラス

**責務:**
- Warren-Cowley SRO パラメータの計算
- 結果の解釈

**主要メソッド:**

1. `__init__(crystal)`: CrystalGeometryオブジェクトを受け取る
2. `calculate_alpha(shell, tolerance)`: αの計算
3. `interpret_alpha(alpha)`: αの物理的解釈

**設計原則:**
- **分離:** 幾何学と物理量計算を分離
- **柔軟性:** 任意の近接殻に対応
- **ロバスト性:** 境界値処理

### 結晶構造タイプの実装

#### Simple Cubic (SC)

```python
if self.structure_type == "SC":
    positions.append(base_pos)  # 1 atom per unit cell
```

- **配位数:** 6
- **充填率:** 52.4%
- **例:** Po (ポロニウム)

#### Body-Centered Cubic (BCC)

```python
elif self.structure_type == "BCC":
    positions.append(base_pos)  # Corner
    positions.append(base_pos + np.array([a/2, a/2, a/2]))  # Center
```

- **配位数:** 8 (第1近接), 6 (第2近接)
- **充填率:** 68.0%
- **例:** Fe, Cr, W

#### Face-Centered Cubic (FCC)

```python
elif self.structure_type == "FCC":
    positions.append(base_pos)  # Corner
    positions.append(base_pos + np.array([a/2, a/2, 0]))  # Face 1
    positions.append(base_pos + np.array([a/2, 0, a/2]))  # Face 2
    positions.append(base_pos + np.array([0, a/2, a/2]))  # Face 3
```

- **配位数:** 12
- **充填率:** 74.0% (最密充填)
- **例:** Al, Cu, Au, Ni

### Data Reduction の原理

#### 原理

アルゴリズムの正しさは、データサイズ $N$ に依存しません。小さなサンプル（$2\times2\times2$）で本質的な挙動を確認できます。

#### 数学的根拠

1. **局所性:** SROパラメータは局所的な相関を測定
2. **統計性:** 十分なサンプリングで統計的性質が収束
3. **境界効果:** 周期境界条件で緩和可能

#### 実装上の利点

- **高速計算:** 原子数が少ない → 計算時間が短い
- **学習効率:** 本質的な挙動を素早く確認
- **反復実験:** 多数のパラメータ組み合わせを試せる

#### 推奨設定

| 目的 | サイズ | 原子数 (FCC) | 計算時間 |
|------|--------|-------------|---------|
| 学習・探索 | 2×2×2 | 32 | < 1秒 |
| 詳細解析 | 3×3×3 | 108 | 数秒 |
| 高精度計算 | 4×4×4 | 256 | 数十秒 |

---

## 使用方法

### インストールと起動

#### 必要な環境

- Python 3.8以上
- 必要なパッケージ（requirements.txtに記載）

#### 起動方法

```bash
cd /path/to/machine-learning
streamlit run crystal_structure_app.py
```

ブラウザが自動的に開き、アプリケーションが表示されます。

### Interactive Mode（インタラクティブモード）

#### 目的

単一の条件で詳細な解析を行います。

#### 使用手順

1. **サイドバーで結晶構造タイプを選択**
   - FCC推奨（最密充填構造）
   - BCC: 体心立方構造
   - SC: 単純立方構造

2. **サイズを選択**
   - 2×2×2推奨（高速計算）
   - 3×3×3: より詳細な解析
   - 4×4×4: 高精度計算

3. **B原子濃度をスライダーで調整**
   - 0.0: 全てA原子
   - 0.5: A:B = 1:1
   - 1.0: 全てB原子

4. **3D可視化で原子配置を確認**
   - マウスで回転: E(3)不変性を体感
   - ズーム: 詳細な配置を観察
   - 青: A原子、赤: B原子

5. **SROパラメータの値と解釈を確認**
   - α ≈ 0: ランダム構造
   - α < 0: 規則構造
   - α > 0: クラスター構造

#### ヒント

- Random Seedを変更して、異なる配置を試す
- 濃度0.5付近でランダム構造を観察
- 濃度0.0, 1.0で境界値動作を確認

### Sweep & Verify Mode（網羅的検証モード）

#### 目的

パラメータ空間を網羅的に探索し、統計的傾向を把握します。

#### 使用手順

1. **サイドバーで「Enable Sweep Mode」をチェック**

2. **検証する結晶構造を選択**
   - 複数選択可能
   - SC, BCC, FCCを比較

3. **濃度の刻み数を設定**
   - 20推奨（バランス）
   - 5-10: 高速探索
   - 30-50: 高解像度

4. **各点での試行回数を設定**
   - 5-10推奨（統計的信頼性）
   - 1: 高速確認
   - 20: 高精度

5. **「Run Validation」ボタンをクリック**

6. **リアルタイムでグラフが更新されるのを観察**
   - 進捗バーで進行状況を確認
   - グラフが動的に更新
   - 完了後、結果テーブルを確認

#### 計算時間の見積もり

計算時間 ≈ (構造数) × (刻み数) × (試行回数) × (単位時間)

例：
- FCC, 20刻み, 5試行 → 約10-20秒
- 3構造, 50刻み, 10試行 → 約5-10分

#### ヒント

- 最初は低解像度（刻み数10, 試行3）で全体像を把握
- 興味深い領域を見つけたら、高解像度で詳細解析
- 複数構造を同時に比較し、配位数の影響を観察

---

## 検証戦略

### 検証マトリクス

以下の項目を必ず確認してください：

| ケース | 設定条件 | 期待される挙動 | 確認事項 |
|--------|---------|---------------|---------|
| **境界値テスト** | 濃度 0.0, 1.0 | エラーなし、α=0 | ゼロ除算が発生しない |
| **ランダム性確認** | 濃度 0.5, 試行10回 | α≈0に収束 | 統計的ゆらぎが小さい |
| **構造依存性** | SC vs FCC | 配位数の違いによる差 | 配位数が大きいほどαの絶対値が小さい |
| **ロバスト性** | 刻み数50, 試行10 | アプリが完走 | メモリエラーが発生しない |

### テスト手順

#### 1. 境界値テスト

```
設定: FCC, 2×2×2, 濃度=0.0
期待: α = 0.0, エラーなし

設定: FCC, 2×2×2, 濃度=1.0
期待: α = 0.0, エラーなし
```

#### 2. ランダム性確認

```
設定: FCC, 2×2×2, 濃度=0.5
試行: Random Seed = 0, 1, 2, ..., 9
期待: αの平均 ≈ 0, 標準偏差 < 0.2
```

#### 3. 構造依存性

```
設定: SC, BCC, FCC, 2×2×2, 濃度=0.5
期待: 
- SC (配位数6): |α| が最大
- BCC (配位数8): |α| が中間
- FCC (配位数12): |α| が最小
```

#### 4. ロバスト性

```
設定: Sweep Mode, FCC, 刻み数50, 試行10
期待: 
- 計算完了（エラーなし）
- 進捗バーが100%に到達
- 結果グラフが表示
```

### デバッグ方法

#### 問題: αの値が常に0になる

**原因:**
- 濃度が0.0または1.0
- 全ての原子が同じ種類

**解決策:**
- 濃度を0.1-0.9の範囲に設定
- Random Seedを変更

#### 問題: Sweep Modeが遅い

**原因:**
- サイズが大きい（3×3×3以上）
- 刻み数・試行回数が多い

**解決策:**
- サイズを2×2×2に設定
- 刻み数を10-20に削減
- 試行回数を3-5に削減

#### 問題: 3D可視化が表示されない

**原因:**
- ブラウザの互換性問題
- Plotlyの読み込みエラー

**解決策:**
- ブラウザを更新（F5）
- Chrome, Firefox, Edgeを使用
- キャッシュをクリア

---

## 参考文献

### 主要論文

1. **Cowley, J. M. (1950).** "An Approximate Theory of Order in Alloys." *Physical Review*, 77(5), 669-675.
   - Warren-Cowley SROパラメータの原典

2. **Warren, B. E. (1969).** *X-Ray Diffraction.* Dover Publications.
   - X線回折による結晶構造解析の古典的名著

3. **de Fontaine, D. (1979).** "Configurational Thermodynamics of Solid Solutions." *Solid State Physics*, 34, 73-274.
   - 合金の配置熱力学の包括的レビュー

### 幾何学的深層学習

4. **Bronstein, M. M., et al. (2021).** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." *arXiv:2104.13478*
   - E(3)不変性と幾何学的深層学習の包括的レビュー

5. **Thomas, N., et al. (2018).** "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds." *arXiv:1802.08219*
   - E(3)等変ニューラルネットワークの基礎

6. **Batzner, S., et al. (2022).** "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." *Nature Communications*, 13, 2453.
   - 材料科学への応用例

### 教科書

7. **Kittel, C. (2004).** *Introduction to Solid State Physics* (8th ed.). Wiley.
   - 結晶構造と対称性の基礎

8. **Ashcroft, N. W., & Mermin, N. D. (1976).** *Solid State Physics.* Holt, Rinehart and Winston.
   - 固体物理学の標準的教科書

### オンラインリソース

- **Materials Project:** https://materialsproject.org/
  - 結晶構造データベース

- **Crystallography Open Database:** http://www.crystallography.net/
  - オープンアクセスの結晶構造データ

- **E3NN (PyTorch):** https://e3nn.org/
  - E(3)等変ニューラルネットワークのライブラリ

---

## 発展的学習トピック

### 1. より高度な秩序パラメータ

- **多体相関関数:** 3体、4体相関の計算
- **動径分布関数 (RDF):** 連続的な距離依存性
- **構造因子 S(q):** 逆格子空間での解析

### 2. 統計力学との接続

- **Ising模型:** 格子上のスピン系との対応
- **Cluster Variation Method (CVM):** より精密な自由エネルギー計算
- **Monte Carlo法:** 熱平衡状態のシミュレーション

### 3. 機械学習への応用

- **Graph Neural Networks (GNN):** 結晶構造の表現学習
- **E(3)等変ネットワーク:** 対称性を保存するニューラルネットワーク
- **Materials Informatics:** 物性予測と材料設計

---

## 付録

### A. 数学記号一覧

| 記号 | 意味 | 例 |
|------|------|-----|
| $\mathbb{R}^3$ | 3次元実数空間 | 位置ベクトル |
| $\mathbb{Z}$ | 整数集合 | 格子点の係数 |
| $\otimes$ | 直積（テンソル積） | $x_i \otimes f_i$ |
| $\bigoplus$ | 直和 | $\bigoplus_{i=1}^{N} s_i$ |
| $\text{span}$ | 生成空間 | $\text{span}(\mathbf{a}, \mathbf{b}, \mathbf{c})$ |
| $SO(3)$ | 3次元特殊直交群 | 回転行列の群 |
| $E(3)$ | 3次元ユークリッド群 | 等長変換の群 |

### B. プログラム構造

```
crystal_structure_app.py
├── CrystalGeometry クラス
│   ├── __init__: 初期化
│   ├── _get_basis_vectors: 基底ベクトル生成
│   ├── _generate_positions: 原子位置生成
│   ├── assign_species: 化学種割り当て
│   ├── get_coordination_number: 配位数取得
│   ├── calculate_neighbor_distances: 距離計算
│   └── get_state_representation: 状態表現
│
├── WarrenCowleySRO クラス
│   ├── __init__: 初期化
│   ├── calculate_alpha: SRO計算
│   └── interpret_alpha: 結果解釈
│
└── main 関数
    ├── Section 1: 数学的定義
    ├── Section 2: Interactive Mode
    ├── Section 3: Sweep & Verify Mode
    ├── Section 4: 検証マトリクス
    └── Section 5: 詳細理論解説
```

### C. よくある質問 (FAQ)

**Q1: なぜ2×2×2が推奨サイズなのですか？**

A: Data Reduction原則に基づき、アルゴリズムの正しさはデータサイズに依存しないため、小さなサンプルで本質的な挙動を確認できます。計算時間を削減し、学習効率を最大化できます。

**Q2: E(3)不変性とは何ですか？**

A: 3次元ユークリッド空間における等長変換（回転・並進・反射）に対して値が変わらない性質です。物理量の本質的な性質を表します。

**Q3: αが負の値になるのはなぜですか？**

A: αが負の値になるのは、A原子の周りにB原子が優先的に配置される規則構造を意味します。これはエンタルピー的に安定な状態です。

**Q4: Sweep Modeで何を確認すべきですか？**

A: 濃度全域でのαの振る舞い、統計的ゆらぎ、構造依存性、境界値での動作を確認してください。

**Q5: 実際の材料研究にどう応用できますか？**

A: 合金の短範囲秩序の定量化、相分離傾向の予測、熱処理条件の最適化などに応用できます。

---

## ライセンス

MIT License

Copyright (c) 2025 Devin AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Document Version:** 1.0.0  
**Last Updated:** 2025-11-24  
**Contact:** https://app.devin.ai/sessions/7a55af1377614a72b91ebebbed2e0c86
