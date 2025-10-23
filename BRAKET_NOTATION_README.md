# ブラケット（Bra-Ket）記法 学習システム

## 概要

このプロジェクトは、量子力学におけるディラックのブラケット記法を学習するための包括的な教育システムです。理論、計算、実装の3つの側面から、若手エンジニアが量子力学の数学的基礎を習得できるように設計されています。

## 特徴

### 1. 完全なPython実装
- **Ket, Bra, Operator クラス**: 量子状態と演算子の完全なオブジェクト指向実装
- **数値計算**: NumPyベースの高速な行列演算
- **型安全**: 明確な型定義とエラーハンドリング

### 2. 教育的コンテンツ
- **Jupyterノートブック**: 理論と実装を統合した対話的チュートリアル
- **豊富な例題**: スピン1/2系、パウリ行列、ベル状態など
- **視覚化**: ブロッホ球による状態の3D表示

### 3. インタラクティブGUI
- **PyQt6アプリケーション**: リアルタイムで状態を操作
- **ブロッホ球可視化**: 量子状態の幾何学的表現
- **期待値計算**: パウリ行列の期待値を即座に表示

### 4. 包括的テスト
- **100以上のユニットテスト**: すべての機能を検証
- **数学的性質の確認**: エルミート性、ユニタリ性、直交性など
- **物理的妥当性**: 確率の保存、不確定性原理など

## ファイル構成

```
machine-learning/
├── braket_notation.py              # コアモジュール（Ket, Bra, Operator クラス）
├── test_braket_notation.py         # ユニットテスト
├── braket_examples.py              # 詳細な例題集
├── braket_gui.py                   # PyQt6 GUIアプリケーション
├── braket_notation_tutorial.ipynb  # Jupyterチュートリアル
└── BRAKET_NOTATION_README.md       # このファイル
```

## インストール

### 必要な依存関係

```bash
pip install numpy matplotlib scipy
```

### オプション（GUI使用時）

```bash
pip install PyQt6
```

### オプション（Jupyter使用時）

```bash
pip install jupyter nbformat
```

## 使い方

### 1. 基本的な使用例

```python
from braket_notation import Ket, Bra, Operator, QuantumStates, PauliMatrices

# 量子状態の作成
psi = QuantumStates.spin_up()
phi = QuantumStates.plus_state()

# 内積の計算
inner_product = psi.bra() * phi
print(f"⟨ψ|φ⟩ = {inner_product}")

# パウリ行列の期待値
sigma_z = PauliMatrices.sigma_z()
expectation = sigma_z.expectation_value(psi)
print(f"⟨σᵤ⟩ = {expectation}")
```

### 2. ブロッホ球の可視化

```python
from braket_notation import BlochSphere
import matplotlib.pyplot as plt

# 状態のリスト
states = [
    QuantumStates.spin_up(),
    QuantumStates.plus_state(),
    QuantumStates.right_circular()
]

labels = ["|↑⟩", "|+⟩", "|R⟩"]

# ブロッホ球にプロット
fig = BlochSphere.plot_bloch_sphere(states, labels)
plt.show()
```

### 3. 例題の実行

```bash
python braket_examples.py
```

これにより、以下の例題が順次実行されます：
- 基本的な量子状態
- パウリ行列の性質
- 期待値と測定
- 射影演算子
- ブロッホ球表現
- 時間発展
- 多粒子系とエンタングルメント
- 不確定性原理

### 4. GUIアプリケーションの起動

```bash
python braket_gui.py
```

GUIでは以下の操作が可能です：
- 量子状態の成分を対話的に入力
- 定義済み状態の選択
- パウリ行列の期待値をリアルタイム計算
- ブロッホ球上での状態の可視化

### 5. Jupyterチュートリアル

```bash
jupyter notebook braket_notation_tutorial.ipynb
```

## 理論的背景

### ブラケット記法とは

ディラックによって導入されたブラケット記法は、量子力学の状態と演算子を表現する標準的な数学的記法です。

#### ケット（Ket）|ψ⟩
量子状態を表す列ベクトル：
```
|ψ⟩ = (ψ₁)
      (ψ₂)
```

#### ブラ（Bra）⟨ψ|
ケットのエルミート共役（複素共役転置）：
```
⟨ψ| = (ψ₁*, ψ₂*)
```

#### 内積 ⟨φ|ψ⟩
二つの状態の内積は複素数：
```
⟨φ|ψ⟩ = φ₁*ψ₁ + φ₂*ψ₂
```

物理的意味：|⟨φ|ψ⟩|² は、状態|ψ⟩を測定して|φ⟩を得る確率

#### 外積 |ψ⟩⟨φ|
演算子を生成：
```
|ψ⟩⟨φ| = (ψ₁)(φ₁*, φ₂*) = (ψ₁φ₁*  ψ₁φ₂*)
          (ψ₂)              (ψ₂φ₁*  ψ₂φ₂*)
```

### パウリ行列

スピン1/2系の基本的な演算子：

```
σₓ = (0  1)    σᵧ = (0  -i)    σᵤ = (1   0)
     (1  0)         (i   0)         (0  -1)
```

性質：
- エルミート: σᵢ† = σᵢ
- 固有値: ±1
- σᵢ² = I（単位行列）
- 交換関係: [σₓ, σᵧ] = 2iσᵤ

### ブロッホ球

スピン1/2の任意の純粋状態は、単位球面上の点として表現できます：

```
|ψ⟩ = cos(θ/2)|↑⟩ + e^(iφ)sin(θ/2)|↓⟩
```

ブロッホベクトル：
```
r⃗ = (⟨σₓ⟩, ⟨σᵧ⟩, ⟨σᵤ⟩)
```

## API リファレンス

### Ket クラス

量子状態ベクトル |ψ⟩ を表現します。

#### メソッド

- `__init__(state)`: 状態ベクトルから初期化
- `normalize()`: 状態を正規化
- `is_normalized()`: 正規化されているか確認
- `bra()`: 対応するブラを返す
- `tensor_product(other)`: テンソル積を計算

#### 演算子

- `+`, `-`: ケットの加減算
- `*`, `/`: スカラー倍
- `@`: テンソル積（`tensor_product`の別名）

### Bra クラス

ブラ ⟨ψ| を表現します。

#### メソッド

- `__init__(state)`: 状態ベクトルから初期化
- `ket()`: 対応するケットを返す

#### 演算子

- `*`: ケットとの内積、または演算子との積

### Operator クラス

量子演算子を表現します。

#### メソッド

- `__init__(matrix)`: 行列から初期化
- `is_hermitian()`: エルミート性を確認
- `is_unitary()`: ユニタリ性を確認
- `dagger()`: エルミート共役を返す
- `eigenvalues_eigenvectors()`: 固有値と固有ベクトルを計算
- `expectation_value(state)`: 期待値を計算
- `commutator(other)`: 交換子 [A, B] を計算
- `anticommutator(other)`: 反交換子 {A, B} を計算

#### 演算子

- `*`: ケットへの作用、演算子の積、スカラー倍
- `+`, `-`: 演算子の加減算

### QuantumStates クラス

定義済みの量子状態を提供します。

#### 静的メソッド

- `spin_up()`: スピン上状態 |↑⟩
- `spin_down()`: スピン下状態 |↓⟩
- `plus_state()`: プラス状態 |+⟩ = (|↑⟩ + |↓⟩)/√2
- `minus_state()`: マイナス状態 |-⟩ = (|↑⟩ - |↓⟩)/√2
- `right_circular()`: 右円偏光 |R⟩
- `left_circular()`: 左円偏光 |L⟩

### PauliMatrices クラス

パウリ行列と関連演算子を提供します。

#### 静的メソッド

- `sigma_x()`: パウリX行列
- `sigma_y()`: パウリY行列
- `sigma_z()`: パウリZ行列
- `identity()`: 単位行列
- `hadamard()`: アダマールゲート
- `phase_gate(phi)`: 位相ゲート

### BlochSphere クラス

ブロッホ球表現のユーティリティ。

#### 静的メソッド

- `state_to_bloch_vector(ket)`: 状態をブロッホベクトルに変換
- `bloch_vector_to_state(vector)`: ブロッホベクトルを状態に変換
- `plot_bloch_sphere(states, labels, title)`: ブロッホ球をプロット

## テスト

すべてのテストを実行：

```bash
python -m pytest test_braket_notation.py -v
```

または標準のunittestで：

```bash
python test_braket_notation.py
```

テストカバレッジ：
- Ketクラス: 正規化、演算、テンソル積
- Braクラス: 内積、変換
- Operatorクラス: エルミート性、ユニタリ性、固有値問題
- 期待値: 実数性、測定確率
- パウリ行列: 交換関係、固有値
- ブロッホ球: 変換、可視化
- 物理的性質: 不確定性原理、完全性関係

## 学習ロードマップ

### 初級（1-2週間）

1. **基本概念の理解**
   - ケットとブラの定義
   - 内積と正規化
   - 直交性

2. **簡単な計算**
   - スピン状態の内積
   - 測定確率の計算
   - 状態の正規化

3. **Pythonでの実装**
   - Ket, Braクラスの使用
   - 基本的な演算

### 中級（2-3週間）

1. **演算子の理解**
   - パウリ行列
   - エルミート演算子
   - 期待値の計算

2. **固有値問題**
   - 固有値と固有ベクトル
   - 測定と状態の崩壊
   - 射影演算子

3. **ブロッホ球**
   - 幾何学的表現
   - 状態の可視化

### 上級（3-4週間）

1. **時間発展**
   - シュレーディンガー方程式
   - ユニタリ発展
   - ブロッホ球上の軌跡

2. **多粒子系**
   - テンソル積
   - エンタングルメント
   - ベル状態

3. **量子情報への応用**
   - 量子ゲート
   - 量子回路
   - 量子アルゴリズム

## 演習問題

### 演習1: 正規直交基底の確認

|↑⟩ と |↓⟩ が正規直交基底であることを確認せよ。

```python
# ヒント: 内積を計算
# ⟨↑|↑⟩ = 1, ⟨↓|↓⟩ = 1, ⟨↑|↓⟩ = 0
```

### 演習2: 射影演算子の冪等性

射影演算子 P = |ψ⟩⟨ψ| が P² = P を満たすことを証明せよ。

```python
# ヒント: 行列の積を計算
```

### 演習3: エルミート演算子の期待値

エルミート演算子の期待値 ⟨ψ|A|ψ⟩ が常に実数であることを示せ。

```python
# ヒント: 複素共役を取る
```

### 演習4: 不確定性原理

σₓ と σᵧ に対する不確定性関係を確認せよ。

```python
# ヒント: ΔA·ΔB ≥ |⟨[A,B]⟩|/2
```

### 演習5: ベル状態の生成

4つのベル状態を生成し、それらが正規直交基底を形成することを確認せよ。

```python
# ヒント: テンソル積を使用
```

## トラブルシューティング

### PyQt6がインストールできない

```bash
# 代替方法
pip install PyQt5
# braket_gui.py の import 文を PyQt5 に変更
```

### matplotlibの3Dプロットが表示されない

```bash
pip install --upgrade matplotlib
```

### Jupyterカーネルが見つからない

```bash
python -m ipykernel install --user
```

## 参考文献

### 書籍

1. J. J. Sakurai, "Modern Quantum Mechanics" (2nd Edition)
   - ディラック記法の標準的な教科書

2. Nielsen & Chuang, "Quantum Computation and Quantum Information"
   - 量子情報理論への応用

3. Griffiths, "Introduction to Quantum Mechanics" (3rd Edition)
   - 初学者向けの丁寧な解説

### オンラインリソース

1. [Qiskit Textbook](https://qiskit.org/textbook/)
   - IBM提供の量子計算教材

2. [Quantum Computing for the Very Curious](https://quantum.country/)
   - 対話的な学習コンテンツ

3. [Wikipedia: Bra-ket notation](https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation)
   - 基本的な定義と歴史

## ライセンス

このプロジェクトは教育目的で作成されました。自由に使用、改変、配布できます。

## 貢献

バグ報告、機能要望、プルリクエストを歓迎します。

## 作成者

Devin AI Assistant
作成日: 2025年10月

## 謝辞

このプロジェクトは、量子力学と量子情報理論の教育を支援するために作成されました。ディラックの記法の優雅さと、Pythonの表現力の組み合わせにより、抽象的な概念を具体的に理解できることを願っています。
