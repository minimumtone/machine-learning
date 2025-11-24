# HEA/CCA機械学習分子動力学シミュレーター 完全マニュアル

## High-Entropy Alloys Machine Learning Molecular Dynamics Simulator - Complete Manual

**バージョン**: 1.0  
**作成日**: 2025年11月24日  
**物理的正確さを最重要視した実装**

---

## 目次

1. [はじめに](#1-はじめに)
2. [理論的背景](#2-理論的背景)
3. [数学的基礎](#3-数学的基礎)
4. [実装の詳細](#4-実装の詳細)
5. [使用方法](#5-使用方法)
6. [検証シナリオ](#6-検証シナリオ)
7. [物理的妥当性の検証](#7-物理的妥当性の検証)
8. [トラブルシューティング](#8-トラブルシューティング)
9. [参考文献](#9-参考文献)

---

## 1. はじめに

### 1.1 本シミュレーターの目的

本アプリケーションは、**ハイエントロピー合金 (High-Entropy Alloys, HEA)** および **複雑組成合金 (Complex Concentrated Alloys, CCA)** の分子動力学シミュレーションを、**機械学習ポテンシャル (Machine Learning Potential, MLP)** を用いて実行するための教育・研究ツールです。

### 1.2 なぜHEA/CCAか？

従来の合金設計では、1-2種類の主要元素に微量の添加元素を加える手法が主流でした。しかし、HEAは**5種類以上の元素を等原子量で混合**することで、従来にない特性を発現します：

- **高強度と高延性の両立**
- **優れた耐熱性**
- **極低温での靭性維持**
- **耐食性・耐酸化性**

### 1.3 なぜ機械学習ポテンシャルか？

多元素系のシミュレーションでは、従来の経験的ポテンシャル（EAM、Tersoffなど）では以下の問題があります：

1. **パラメータ数の爆発**: N元素系でO(N²)〜O(N³)のパラメータが必要
2. **パラメータ不足**: 多くの元素組み合わせでパラメータが存在しない
3. **精度の限界**: 複雑な化学環境を表現できない

機械学習ポテンシャル（特にMACE）は、これらの問題を解決します：

- **第一原理計算（DFT）に近い精度**
- **多元素系への汎用性**
- **経験的ポテンシャル並みの計算速度**

---

## 2. 理論的背景

### 2.1 ハイエントロピー合金の定義

#### 2.1.1 組成の定義

HEAは以下の条件を満たす合金です：

- **主要元素が5種類以上**
- **各元素の組成が5-35 at.%**（等原子量または準等原子量）

代表例：
- **Cantor合金**: CoCrFeMnNi（各20 at.%）
- **Refractory HEA**: TiZrNbHfTa

#### 2.1.2 配置エントロピー

HEAの安定化メカニズムの中心となるのが**配置エントロピー** (Configurational Entropy) です。

**定義式**:

$$S_{conf} = -R \sum_{i=1}^N c_i \ln c_i$$

ここで：
- $R = 8.314$ J/(mol·K): 気体定数
- $c_i$: 元素$i$のモル分率
- $N$: 構成元素数

**等原子量の場合** ($c_i = 1/N$):

$$S_{conf} = R \ln N$$

**数値例**:

| 元素数 N | $S_{conf}$ (J/mol·K) | 備考 |
|---------|---------------------|------|
| 1       | 0                   | 純金属 |
| 2       | 5.76                | 二元合金 |
| 3       | 9.13                | 三元合金 |
| 4       | 11.53               | 四元合金 |
| 5       | 13.38               | HEA（最小） |
| 6       | 14.90               | HEA |

**物理的意味**:

ギブス自由エネルギー:

$$G = H - TS$$

- $H$: エンタルピー（混合エンタルピー $\Delta H_{mix}$ は通常正、不利）
- $-TS$: エントロピー項（常に負、有利）

高温では $T\Delta S_{conf}$ が大きくなり、$\Delta H_{mix}$ を上回ることで固溶体が安定化されます。これを**高エントロピー効果**と呼びます。

### 2.2 局所格子歪み (Lattice Distortion)

#### 2.2.1 原子半径の違い

HEAでは、異なる原子半径を持つ元素が隣接するため、原子位置が理想的な格子点からずれます。

**代表的な元素の原子半径**（金属結合半径、Å）:

| 元素 | 原子半径 | 相対差（Niを基準） |
|-----|---------|------------------|
| Ni  | 1.24    | 0%               |
| Co  | 1.25    | +0.8%            |
| Fe  | 1.26    | +1.6%            |
| Mn  | 1.27    | +2.4%            |
| Cr  | 1.28    | +3.2%            |
| Cu  | 1.28    | +3.2%            |
| V   | 1.34    | +8.1%            |
| Al  | 1.43    | +15.3%           |
| Ti  | 1.47    | +18.5%           |

#### 2.2.2 格子歪みの定量化

**変位ベクトル**:

$$\Delta \boldsymbol{r}_i = \boldsymbol{r}_i - \boldsymbol{R}_i$$

- $\boldsymbol{r}_i$: 実際の原子位置
- $\boldsymbol{R}_i$: 理想的な格子点位置

**格子歪みパラメータ** (δ):

$$\delta = \sqrt{\sum_{i=1}^N c_i \left(1 - \frac{r_i}{\bar{r}}\right)^2}$$

ここで：
- $r_i$: 元素$i$の原子半径
- $\bar{r} = \sum_{i=1}^N c_i r_i$: 平均原子半径

**経験則**:

- $\delta < 3\%$: 固溶体形成が容易
- $3\% < \delta < 6\%$: 固溶体形成可能だが歪みエネルギー大
- $\delta > 6\%$: 相分離やアモルファス化の可能性

**Cantor合金の例**:

$$\bar{r} = \frac{1.25 + 1.28 + 1.26 + 1.27 + 1.24}{5} = 1.26 \text{ Å}$$

$$\delta = \sqrt{\frac{1}{5}\left[\left(\frac{1.25-1.26}{1.26}\right)^2 + \cdots\right]} \approx 1.2\%$$

→ 固溶体形成が容易（実験結果と一致）

#### 2.2.3 格子歪みの効果

1. **強度向上**: 転位の移動が阻害される（固溶強化）
2. **拡散の遅延**: 複雑なエネルギー地形により拡散係数が低下
3. **熱伝導率の低下**: フォノン散乱の増大

### 2.3 HEAの4つの主要効果

#### 1. 高エントロピー効果 (High-Entropy Effect)

配置エントロピーによる固溶体の安定化。

#### 2. 格子歪み効果 (Severe Lattice Distortion Effect)

原子半径差による強度向上と拡散抑制。

#### 3. 遅い拡散効果 (Sluggish Diffusion Effect)

複雑な化学環境により、拡散係数が純金属の1/10〜1/100に低下。

**拡散係数の温度依存性**:

$$D = D_0 \exp\left(-\frac{Q}{RT}\right)$$

HEAでは活性化エネルギー $Q$ が大きい。

#### 4. カクテル効果 (Cocktail Effect)

個々の元素にはない新しい性質の発現。

---

## 3. 数学的基礎

### 3.1 分子動力学の基礎方程式

#### 3.1.1 ニュートンの運動方程式

$$m_i \frac{d^2 \boldsymbol{r}_i}{dt^2} = \boldsymbol{F}_i$$

ここで：
- $m_i$: 原子$i$の質量
- $\boldsymbol{r}_i$: 原子$i$の位置ベクトル
- $\boldsymbol{F}_i$: 原子$i$に働く力

#### 3.1.2 力の計算

ポテンシャルエネルギー $E_{total}$ から：

$$\boldsymbol{F}_i = -\frac{\partial E_{total}}{\partial \boldsymbol{r}_i}$$

### 3.2 Langevin動力学

本シミュレーターでは、**Langevin動力学**を採用しています。

**運動方程式**:

$$m_i \frac{d^2 \boldsymbol{r}_i}{dt^2} = \boldsymbol{F}_i - \gamma m_i \frac{d\boldsymbol{r}_i}{dt} + \boldsymbol{\xi}_i(t)$$

ここで：
- $\gamma$: 摩擦係数（熱浴との結合強度）
- $\boldsymbol{\xi}_i(t)$: ランダム力（ガウス白色雑音）

**ランダム力の性質**:

$$\langle \boldsymbol{\xi}_i(t) \rangle = 0$$

$$\langle \boldsymbol{\xi}_i(t) \cdot \boldsymbol{\xi}_j(t') \rangle = 2\gamma m_i k_B T \delta_{ij} \delta(t - t')$$

これは**揺動散逸定理** (Fluctuation-Dissipation Theorem) を満たします。

**利点**:

1. **正準集団（NVT）の実現**: 指定温度での熱平衡
2. **速い緩和**: 初期状態から平衡状態への移行が速い
3. **実験条件との対応**: 一定温度での測定に対応

### 3.3 時間積分アルゴリズム

**Velocity Verlet法**（Langevin動力学用に修正）:

1. 速度の半ステップ更新:
   $$\boldsymbol{v}(t + \Delta t/2) = \boldsymbol{v}(t) + \frac{\boldsymbol{F}(t)}{m} \frac{\Delta t}{2}$$

2. 位置の更新:
   $$\boldsymbol{r}(t + \Delta t) = \boldsymbol{r}(t) + \boldsymbol{v}(t + \Delta t/2) \Delta t$$

3. 力の再計算:
   $$\boldsymbol{F}(t + \Delta t) = -\frac{\partial E}{\partial \boldsymbol{r}}(t + \Delta t)$$

4. 速度の完全更新:
   $$\boldsymbol{v}(t + \Delta t) = \boldsymbol{v}(t + \Delta t/2) + \frac{\boldsymbol{F}(t + \Delta t)}{m} \frac{\Delta t}{2}$$

**タイムステップの選択**:

- 通常 1-2 fs（$10^{-15}$ 秒）
- 原子振動の周期（〜10 fs）の約1/10
- 安定性条件: $\Delta t < \frac{2}{\omega_{max}}$（$\omega_{max}$: 最大振動数）

### 3.4 機械学習ポテンシャル (MACE)

#### 3.4.1 基本原理

全エネルギーを原子ごとのエネルギーの和として表現:

$$E_{total} = \sum_{i=1}^{N_{atoms}} E_i(\text{Environment}_i)$$

ここで $E_i$ はニューラルネットワークで、$\text{Environment}_i$ は原子$i$の局所環境を表す記述子です。

#### 3.4.2 記述子の構成

**入力情報**:

1. **原子種**: 各元素の one-hot エンコーディング
2. **原子間距離**: カットオフ半径内の隣接原子との距離
3. **結合角**: 3体相関
4. **多体相関**: 高次の幾何学的特徴

**等変性** (Equivariance):

記述子は以下の対称性を満たす必要があります：

- **並進不変性**: 系全体を平行移動しても不変
- **回転不変性**: 系全体を回転しても不変
- **置換不変性**: 同種原子の交換に対して不変

#### 3.4.3 MACEの特徴

**MACE (Multi-Atomic Cluster Expansion)** は、等変メッセージパッシングニューラルネットワークです。

**利点**:

1. **高精度**: DFT計算に匹敵する精度
2. **汎用性**: 多元素系に対応
3. **効率性**: 経験的ポテンシャルと同程度の速度
4. **データ効率**: 少ない訓練データで高精度

**MACE-MP-0**:

- Materials Project のデータで訓練
- 89元素に対応
- 様々な結晶構造・化学環境をカバー

#### 3.4.4 力の計算

自動微分により効率的に計算:

$$\boldsymbol{F}_i = -\frac{\partial E_{total}}{\partial \boldsymbol{r}_i}$$

PyTorchの自動微分機能を利用。

### 3.5 熱力学量の計算

#### 3.5.1 温度

瞬間温度は運動エネルギーから計算:

$$T = \frac{2}{3N k_B} \sum_{i=1}^N \frac{1}{2} m_i v_i^2$$

ここで：
- $N$: 原子数
- $k_B = 1.380649 \times 10^{-23}$ J/K: ボルツマン定数

#### 3.5.2 圧力

ビリアル定理より:

$$P = \frac{N k_B T}{V} + \frac{1}{3V} \sum_{i=1}^N \boldsymbol{r}_i \cdot \boldsymbol{F}_i$$

#### 3.5.3 比熱

エネルギー揺らぎから:

$$C_V = \frac{\langle E^2 \rangle - \langle E \rangle^2}{k_B T^2}$$

---

## 4. 実装の詳細

### 4.1 HEA構造生成アルゴリズム

#### 4.1.1 FCC格子の生成

```python
def create_hea_structure(elements, size=(3, 3, 3), lattice_constant=3.52):
    # 1. ベースとなる単元素FCC構造を作成
    atoms = bulk("Cu", crystalstructure="fcc", a=lattice_constant, cubic=True) * size
    
    # 2. 原子総数を取得
    n_atoms = len(atoms)
    
    # 3. 元素リストを作成（等原子量）
    n_elements = len(elements)
    symbols = []
    for i in range(n_atoms):
        symbols.append(elements[i % n_elements])
    
    # 4. ランダムに混ぜる（配置エントロピーの実現）
    random.shuffle(symbols)
    
    # 5. Atomsオブジェクトに適用
    atoms.set_chemical_symbols(symbols)
    
    return atoms
```

**ポイント**:

1. **ベース格子**: Cuの格子定数（3.615 Å）を基準として使用
2. **等原子量**: 各元素が均等に分布するようリストを作成
3. **ランダム配置**: `random.shuffle()` で配置エントロピーを実現
4. **格子定数**: 平均的な値（3.52 Å）を使用、MLPが最適化

#### 4.1.2 格子定数の選択

Cantor合金の実験値: 約3.60 Å

各元素のFCC格子定数:
- Co: 3.545 Å
- Cr: 3.615 Å (BCC→FCC換算)
- Fe: 3.647 Å (BCC→FCC換算)
- Mn: 3.860 Å
- Ni: 3.524 Å

平均: 約3.64 Å

本実装では3.52 Åを使用（やや小さめ）し、MD緩和で最適化させます。

### 4.2 MACE計算機の設定

```python
from mace.calculators import mace_mp

calc = mace_mp(
    model="small",           # 軽量モデル（medium, largeも選択可）
    dispersion=False,        # 分散力補正なし
    default_dtype="float32", # 単精度（速度優先）
    device="cpu"             # CPUモード
)
atoms.calc = calc
```

**パラメータの意味**:

- `model="small"`: MACE-MP-0-small（パラメータ数が少ない）
- `dispersion=False`: van der Waals補正なし（金属では不要）
- `default_dtype="float32"`: 単精度浮動小数点（倍精度より高速）
- `device="cpu"`: CPU実行（GPUがない環境でも動作）

### 4.3 MD初期化

```python
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units

# 初期速度の設定（Maxwell-Boltzmann分布）
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

# Langevin動力学の設定
dyn = Langevin(
    atoms,
    timestep=timestep * units.fs,      # タイムステップ（fs → 内部単位）
    temperature_K=temperature,          # 目標温度（K）
    friction=friction / units.fs        # 摩擦係数（1/fs → 内部単位）
)
```

**初期速度の重要性**:

Maxwell-Boltzmann分布に従う速度を与えることで、系が指定温度の熱平衡状態から開始します。

$$P(v) \propto \exp\left(-\frac{mv^2}{2k_BT}\right)$$

### 4.4 3D可視化

```python
def atoms_to_py3dmol(atoms, show_cell=True):
    # XYZ形式の文字列を生成
    xyz_str = f"{len(atoms)}\n\n"
    for atom in atoms:
        pos = atom.position
        xyz_str += f"{atom.symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
    
    # Py3Dmolビューアを作成
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_str, 'xyz')
    
    # 元素ごとの色設定（CPK配色）
    colors = get_element_colors()
    for element, color in colors.items():
        view.setStyle({'elem': element}, {'sphere': {'color': color, 'radius': 0.5}})
    
    view.zoomTo()
    return view
```

**CPK配色** (Corey-Pauling-Koltun):

化学で標準的に使用される元素の色：
- Co: ピンク
- Cr: 青灰色
- Fe: オレンジ赤
- Mn: 紫
- Ni: 緑

これにより、元素の分布が視覚的に明確になります。

---

## 5. 使用方法

### 5.1 基本的な使用フロー

#### ステップ1: 構造生成

1. **「構造生成」タブ**を開く
2. 構成元素を選択
   - デフォルト: CoCrFeMnNi（Cantor合金）
   - 推奨: 5種類以上
3. 格子定数を設定
   - デフォルト: 3.52 Å
   - 範囲: 3.0-4.5 Å
4. システムサイズを選択
   - 推奨: 3×3×3（108原子）
   - 小: 2×2×2（32原子）- 高速テスト用
   - 大: 4×4×4（256原子）- 詳細解析用
5. **「構造を生成」ボタン**をクリック
6. 3D可視化で構造を確認
   - 元素ごとに色分けされている
   - セルの境界が表示される
   - マウスで回転・ズーム可能

#### ステップ2: MDシミュレーション

1. **「MDシミュレーション」タブ**を開く
2. 温度を設定
   - 室温: 300 K
   - 高温: 1000-1500 K
   - 極低温: 100-200 K
3. タイムステップを設定
   - 推奨: 1.0 fs
   - 範囲: 0.5-2.0 fs
4. ステップ数を設定
   - 初回: 50-100ステップ
   - 詳細: 200-500ステップ
5. 摩擦係数を設定
   - デフォルト: 0.002 /fs
   - 範囲: 0.001-0.01 /fs
6. ポテンシャルを選択
   - **推奨**: MACE（高精度）
   - 代替: EMT（高速、限定的）
7. **「シミュレーション開始」ボタン**をクリック
8. 進行状況を確認
   - プログレスバー
   - リアルタイムのエネルギー・温度表示
9. 完了後、結果グラフを確認
   - ポテンシャルエネルギーの時間変化
   - 温度の時間変化

#### ステップ3: 結果分析

1. **「結果分析」タブ**を開く
2. フレームスライダーで時間発展を確認
   - 初期状態（フレーム0）
   - 中間状態
   - 最終状態（フレーム最大）
3. 構造パラメータを確認
   - RMS変位
   - 最大変位
   - エネルギー・温度
4. 変位分析グラフを確認
   - 元素ごとの変位分布（Box plot）
   - 格子歪みの定量評価
5. データをエクスポート
   - 軌跡データ（CSV）
   - 最終構造（XYZ）

### 5.2 推奨設定

#### 初めての方

```
元素: Co, Cr, Fe, Mn, Ni（Cantor合金）
サイズ: 3×3×3（108原子）
温度: 300 K
タイムステップ: 1.0 fs
ステップ数: 100
摩擦係数: 0.002 /fs
ポテンシャル: MACE
```

**期待される計算時間**: 5-10分（一般的なPC）

#### 高速テスト

```
元素: Ni, Cu, Al（EMT対応元素）
サイズ: 2×2×2（32原子）
温度: 300 K
タイムステップ: 1.0 fs
ステップ数: 50
摩擦係数: 0.002 /fs
ポテンシャル: EMT
```

**期待される計算時間**: 10-30秒

#### 詳細解析

```
元素: Co, Cr, Fe, Mn, Ni
サイズ: 3×3×3（108原子）
温度: 300 K
タイムステップ: 1.0 fs
ステップ数: 500
摩擦係数: 0.002 /fs
ポテンシャル: MACE
```

**期待される計算時間**: 30-60分

---

## 6. 検証シナリオ

### 6.1 ケースA: 純金属 vs HEA の比較

**目的**: HEA特有の格子歪みを観察する

**手順**:

1. **純Niのシミュレーション**
   - 元素: Ni のみ
   - 温度: 300 K
   - ステップ: 100
   - ポテンシャル: MACE

2. **Cantor合金のシミュレーション**
   - 元素: Co, Cr, Fe, Mn, Ni
   - 温度: 300 K
   - ステップ: 100
   - ポテンシャル: MACE

3. **比較項目**
   - RMS変位
   - 元素ごとの変位分布
   - エネルギー揺らぎ

**期待される結果**:

| 項目 | 純Ni | Cantor合金 |
|------|------|-----------|
| RMS変位 | 小（均一） | 大（不均一） |
| 変位分布 | 単一ピーク | 元素ごとに異なる |
| エネルギー揺らぎ | 小 | 大 |

**物理的解釈**:

- **純Ni**: すべての原子が同じ環境 → 均一な振動
- **Cantor合金**: 原子ごとに異なる環境 → 不均一な振動（格子歪み）

### 6.2 ケースB: 高温安定性の確認

**目的**: HEAの高温での構造安定性を確認する

**手順**:

1. **300 Kでのシミュレーション**
   - 元素: Co, Cr, Fe, Mn, Ni
   - 温度: 300 K
   - ステップ: 200

2. **1000 Kでのシミュレーション**
   - 同じ構造
   - 温度: 1000 K
   - ステップ: 200

3. **1500 Kでのシミュレーション**
   - 同じ構造
   - 温度: 1500 K
   - ステップ: 200

4. **比較項目**
   - RMS変位の温度依存性
   - 構造の保持（FCC構造が維持されるか）
   - エネルギーの安定性

**期待される結果**:

| 温度 | RMS変位 | 構造 | 備考 |
|------|---------|------|------|
| 300 K | 小 | FCC維持 | 室温 |
| 1000 K | 中 | FCC維持 | 高温でも安定 |
| 1500 K | 大 | FCC維持または部分的崩壊 | 融点に近い |

**物理的解釈**:

- **カクテル効果**: 高温でも構造が粘り強く維持される
- **遅い拡散**: 原子の移動が抑制される
- **融点**: Cantor合金の融点は約1400-1500 K

### 6.3 ケースC: 不安定な組み合わせ

**目的**: 相分離やアモルファス化の観察

**手順**:

1. **原子半径差の大きい元素を選択**
   - 元素: Al, Ti, Cu, Ni, Co
   - Al: 1.43 Å（+15%）
   - Ti: 1.47 Å（+19%）
   - Cu: 1.28 Å（+3%）
   - Ni: 1.24 Å（基準）
   - Co: 1.25 Å（+1%）

2. **格子歪みパラメータを計算**
   $$\delta \approx 6.5\%$$
   → 固溶体形成が困難な領域

3. **高温でシミュレーション**
   - 温度: 1000 K
   - ステップ: 200-500

4. **観察項目**
   - 構造の変化
   - 相分離の兆候
   - アモルファス化

**期待される結果**:

- **初期**: FCC構造から開始
- **中期**: 格子歪みが増大、局所的な構造変化
- **後期**: 可能性
  1. 相分離（Al-Ti richとCu-Ni-Co richに分離）
  2. アモルファス化（結晶構造の崩壊）
  3. 準安定状態の維持

**物理的解釈**:

- **格子歪みエネルギー**: $\delta > 6\%$ で非常に大きい
- **エントロピー項**: 高温でも歪みエネルギーを補償できない
- **実験との対応**: このような組み合わせは実際にHEAを形成しにくい

---

## 7. 物理的妥当性の検証

### 7.1 エネルギー保存則

**断熱系（NVE）での検証**:

理論的には、外部との熱交換がない場合、全エネルギー $E_{total} = E_{kinetic} + E_{potential}$ は保存されるべきです。

**許容範囲**:

$$\frac{\Delta E}{E} < 0.01 \quad (1\%)$$

**本シミュレーター**:

Langevin動力学（NVT）を使用しているため、エネルギーは厳密には保存されません（熱浴との交換がある）。しかし、平均エネルギーは安定しているべきです。

### 7.2 温度制御

**目標温度との偏差**:

$$\frac{|T_{avg} - T_{target}|}{T_{target}} < 0.05 \quad (5\%)$$

**確認方法**:

1. MDシミュレーション実行後、温度プロットを確認
2. 平均温度を計算
3. 目標温度との差を評価

**調整方法**:

- 偏差が大きい場合: 摩擦係数を調整
  - 温度が低い: 摩擦係数を小さく
  - 温度が高い: 摩擦係数を大きく

### 7.3 構造安定性

**異常な原子間距離のチェック**:

最近接原子間距離 $d_{min}$ が物理的に妥当な範囲にあるか確認。

**許容範囲**:

$$2.0 \text{ Å} < d_{min} < 3.5 \text{ Å}$$

- $d_{min} < 2.0$ Å: 原子が異常に接近（計算エラーの可能性）
- $d_{min} > 3.5$ Å: 構造が崩壊（融解または相分離）

### 7.4 格子歪みの文献値との比較

**Cantor合金の実験値**:

- 格子定数: 3.60 Å
- 格子歪みパラメータ: δ ≈ 1.2%

**シミュレーション結果の確認**:

1. 最終構造の格子定数を測定
2. 格子歪みパラメータを計算
3. 文献値と比較

**許容範囲**:

- 格子定数: ±5%
- 格子歪みパラメータ: ±20%

### 7.5 計算精度の検証

**MACEの精度**:

- DFTとの比較: エネルギー誤差 < 10 meV/atom
- 力の誤差: < 0.1 eV/Å

**本シミュレーターでの確認**:

1. 同じ構造でDFT計算を実行（外部ツール）
2. MACEの結果と比較
3. 誤差を評価

---

## 8. トラブルシューティング

### 8.1 シミュレーションが遅い

**症状**: 計算時間が非常に長い（30分以上）

**原因と対策**:

1. **システムサイズが大きすぎる**
   - 対策: サイズを小さくする（2×2×2）

2. **ステップ数が多すぎる**
   - 対策: ステップ数を減らす（50ステップ）

3. **MACEの計算コスト**
   - 対策: EMTポテンシャルを試す（対応元素のみ）

4. **CPUの性能**
   - 対策: より高性能なマシンを使用

### 8.2 エラーが発生する

**症状**: "Calculator not supported" などのエラー

**原因と対策**:

1. **元素がMACEでサポートされていない**
   - 確認: MACE-MP-0は89元素に対応
   - 対策: サポートされている元素を選択

2. **EMTで非対応元素を使用**
   - EMT対応元素: Cu, Ag, Au, Ni, Pd, Pt, Al
   - 対策: 対応元素のみ選択、またはMACEを使用

3. **メモリ不足**
   - 対策: システムサイズを小さくする

4. **依存パッケージの問題**
   - 対策: 必要なパッケージを再インストール
   ```bash
   pip install --upgrade ase mace-torch
   ```

### 8.3 温度が安定しない

**症状**: 温度が目標値から大きくずれる、または発散する

**原因と対策**:

1. **摩擦係数が小さすぎる**
   - 対策: 摩擦係数を大きくする（0.005-0.01 /fs）

2. **タイムステップが大きすぎる**
   - 対策: タイムステップを小さくする（0.5 fs）

3. **平衡化時間が不足**
   - 対策: ステップ数を増やす（200-500ステップ）

4. **初期構造が不適切**
   - 対策: 格子定数を調整、または構造を再生成

### 8.4 構造が崩壊する

**症状**: 原子が異常に移動する、または結晶構造が失われる

**原因と対策**:

1. **温度が高すぎる（融点以上）**
   - Cantor合金の融点: 約1400-1500 K
   - 対策: 温度を下げる（< 1000 K）

2. **元素の組み合わせが不適切**
   - 格子歪みパラメータ δ > 6%
   - 対策: より相性の良い元素を選択

3. **タイムステップが大きすぎる**
   - 対策: タイムステップを小さくする（0.5 fs）

4. **物理的に正しい挙動の可能性**
   - 相分離やアモルファス化は実際に起こりうる
   - 対策: これは「エラー」ではなく「観察結果」として記録

### 8.5 可視化が表示されない

**症状**: 3D構造が表示されない

**原因と対策**:

1. **ブラウザの互換性**
   - 対策: Chrome、Firefox、Edgeなどの最新ブラウザを使用

2. **JavaScriptが無効**
   - 対策: JavaScriptを有効にする

3. **ネットワーク接続**
   - 対策: インターネット接続を確認（CDNからライブラリを読み込むため）

---

## 9. 参考文献

### 9.1 HEAの基礎

1. Yeh, J. W., Chen, S. K., Lin, S. J., Gan, J. Y., Chin, T. S., Shun, T. T., ... & Chang, S. Y. (2004). "Nanostructured high-entropy alloys with multiple principal elements: novel alloy design concepts and outcomes." *Advanced Engineering Materials*, 6(5), 299-303.

2. Miracle, D. B., & Senkov, O. N. (2017). "A critical review of high entropy alloys and related concepts." *Acta Materialia*, 122, 448-511.

3. George, E. P., Raabe, D., & Ritchie, R. O. (2019). "High-entropy alloys." *Nature Reviews Materials*, 4(8), 515-534.

4. Zhang, Y., Zuo, T. T., Tang, Z., Gao, M. C., Dahmen, K. A., Liaw, P. K., & Lu, Z. P. (2014). "Microstructures and properties of high-entropy alloys." *Progress in Materials Science*, 61, 1-93.

### 9.2 機械学習ポテンシャル

5. Batatia, I., Kovacs, D. P., Simm, G., Ortner, C., & Csányi, G. (2022). "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." *Advances in Neural Information Processing Systems*, 35, 11423-11436.

6. Behler, J., & Parrinello, M. (2007). "Generalized neural-network representation of high-dimensional potential-energy surfaces." *Physical Review Letters*, 98(14), 146401.

7. Schütt, K. T., Sauceda, H. E., Kindermans, P. J., Tkatchenko, A., & Müller, K. R. (2018). "SchNet–A deep learning architecture for molecules and materials." *The Journal of Chemical Physics*, 148(24), 241722.

### 9.3 分子動力学

8. Frenkel, D., & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.

9. Allen, M. P., & Tildesley, D. J. (2017). *Computer Simulation of Liquids*. Oxford University Press.

10. Leach, A. R. (2001). *Molecular Modelling: Principles and Applications*. Pearson Education.

### 9.4 計算材料科学

11. Martin, R. M. (2004). *Electronic Structure: Basic Theory and Practical Methods*. Cambridge University Press.

12. Kalidindi, S. R., & De Graef, M. (2015). "Materials data science: current status and future outlook." *Annual Review of Materials Research*, 45, 171-193.

### 9.5 ASEとPython

13. Larsen, A. H., Mortensen, J. J., Blomqvist, J., Castelli, I. E., Christensen, R., Dułak, M., ... & Jacobsen, K. W. (2017). "The atomic simulation environment—a Python library for working with atoms." *Journal of Physics: Condensed Matter*, 29(27), 273002.

### 9.6 Cantor合金の実験研究

14. Gludovatz, B., Hohenwarter, A., Catoor, D., Chang, E. H., George, E. P., & Ritchie, R. O. (2014). "A fracture-resistant high-entropy alloy for cryogenic applications." *Science*, 345(6201), 1153-1158.

15. Otto, F., Dlouhý, A., Somsen, C., Bei, H., Eggeler, G., & George, E. P. (2013). "The influences of temperature and microstructure on the tensile properties of a CoCrFeMnNi high-entropy alloy." *Acta Materialia*, 61(15), 5743-5755.

---

## 付録A: 物理定数

| 定数 | 記号 | 値 | 単位 |
|------|------|-----|------|
| ボルツマン定数 | $k_B$ | 1.380649 × 10⁻²³ | J/K |
| 気体定数 | $R$ | 8.314 | J/(mol·K) |
| アボガドロ数 | $N_A$ | 6.022 × 10²³ | 1/mol |
| 電子ボルト | eV | 1.602 × 10⁻¹⁹ | J |
| 原子質量単位 | u | 1.661 × 10⁻²⁷ | kg |
| フェムト秒 | fs | 10⁻¹⁵ | s |
| オングストローム | Å | 10⁻¹⁰ | m |

---

## 付録B: 元素データ

### 原子質量（u）

| 元素 | 記号 | 原子質量 |
|------|------|---------|
| アルミニウム | Al | 26.98 |
| コバルト | Co | 58.93 |
| クロム | Cr | 52.00 |
| 銅 | Cu | 63.55 |
| 鉄 | Fe | 55.85 |
| マンガン | Mn | 54.94 |
| ニッケル | Ni | 58.69 |
| チタン | Ti | 47.87 |
| バナジウム | V | 50.94 |

### 融点（K）

| 元素 | 融点 |
|------|------|
| Al | 933 |
| Co | 1768 |
| Cr | 2180 |
| Cu | 1358 |
| Fe | 1811 |
| Mn | 1519 |
| Ni | 1728 |
| Ti | 1941 |
| V | 2183 |

---

## 付録C: よくある質問（FAQ）

### Q1: HEAとCCAの違いは？

**A**: 厳密な定義の違いです。
- **HEA**: 配置エントロピーが高い（$S_{conf} > 1.5R$、つまり5元素以上）
- **CCA**: 複数の主要元素を含むが、必ずしも5元素以上でない

実用上、ほぼ同義で使われることが多いです。

### Q2: なぜFCC構造なのか？

**A**: Cantor合金など、多くの代表的なHEAがFCC構造を取るためです。
ただし、HEAはBCC、HCPなど他の構造も取りえます。

### Q3: 実験との時間スケールの違いは？

**A**: 
- **シミュレーション**: ピコ秒（ps）〜ナノ秒（ns）
- **実験**: 秒〜時間〜日

この差は現在の計算機では埋められません。しかし、短時間の原子レベルの挙動を観察することで、長時間の性質を推測できます。

### Q4: より大きなシステムをシミュレートするには？

**A**: 
1. **GPU版MACEを使用**: `device="cuda"`（NVIDIA GPU必要）
2. **並列化**: 複数のCPUコアを活用
3. **クラスター計算**: HPCシステムを利用

### Q5: 他のMLポテンシャルは使えるか？

**A**: はい。ASEは様々な計算機に対応しています：
- **CHGNet**: 結晶構造予測に強い
- **M3GNet**: Materials Projectベース
- **NequIP**: 等変ニューラルネットワーク
- **DeePMD**: 大規模系に適している

計算機の設定を変更するだけで使用可能です。

---

## 付録D: 発展的なトピック

### D.1 相図の予測

より長時間のシミュレーションと自由エネルギー計算により、HEAの相図を予測できます。

**手法**:
- Umbrella sampling
- Metadynamics
- Thermodynamic integration

### D.2 機械的性質の計算

応力-歪み曲線を計算し、弾性定数や降伏強度を予測できます。

**手法**:
- Uniaxial tension test
- Shear deformation
- Nanoindentation simulation

### D.3 拡散係数の計算

平均二乗変位（MSD）から拡散係数を計算できます。

$$D = \lim_{t \to \infty} \frac{1}{6t} \langle |\boldsymbol{r}(t) - \boldsymbol{r}(0)|^2 \rangle$$

### D.4 動径分布関数（RDF）

原子間距離の分布を解析し、局所構造を定量化できます。

$$g(r) = \frac{1}{4\pi r^2 \rho N} \sum_{i=1}^N \sum_{j \neq i} \delta(r - r_{ij})$$

### D.5 機械学習による物性予測

シミュレーション結果を訓練データとして、機械学習モデルで物性を予測できます。

**例**:
- 組成 → 格子定数
- 組成 → 弾性定数
- 組成 → 融点

---

## おわりに

本マニュアルは、HEA/CCA機械学習分子動力学シミュレーターの包括的なガイドです。

**重要なポイント**:

1. **物理的正確さ**: 本シミュレーターは、最新の機械学習ポテンシャル（MACE）を使用し、物理的に妥当な結果を提供します。

2. **教育目的**: 複雑な多元素系の挙動を視覚的に理解できます。

3. **研究への応用**: 実験前のスクリーニングや、実験結果の解釈に活用できます。

4. **限界の認識**: シミュレーションの時間・空間スケールの限界を理解し、結果を適切に解釈することが重要です。

**今後の展望**:

- より大規模なシミュレーション
- 長時間スケールの現象（相分離、析出など）
- 機械的性質の予測
- 実験データとの統合

HEAは材料科学の最前線であり、本シミュレーターがその理解と発展に貢献することを期待します。

---

**連絡先・フィードバック**:

本マニュアルに関する質問、フィードバック、改善提案は歓迎します。

**引用**:

本シミュレーターを研究で使用する場合、MACEとASEの論文を引用してください。

---

**バージョン履歴**:

- v1.0 (2025-11-24): 初版リリース

---

**ライセンス**:

本ソフトウェアは教育・研究目的で自由に使用できます。

---

**謝辞**:

- MACE開発チーム
- ASE開発チーム
- Materials Project
- HEAコミュニティ

---

**END OF MANUAL**
