# Co系超合金拡散解析アプリケーション

## 概要

このアプリケーションは、Lindwall et al. (2024)の論文「Development of a Diffusion Mobility Database for Co-based Superalloys」に基づいて、Co系超合金の拡散現象を解析するための統合ツールです。

論文データの抽出、拡散方程式の数値解法、高度な可視化機能を一つのStreamlitアプリケーションにまとめています。

## 機能

### 1. 📊 論文データ抽出

- **Figure 19**: Co-Al-Cr / Ni-Al-Co 拡散対（1100°C、48時間）
- **Figure 20**: Co-Al-Cr / Ni-Al-Cr-Ti 拡散対（1100°C、72時間）

各図から抽出されたデータ：
- 距離に対する質量分率プロファイル
- Co, Ni, Cr, Al, Ti の濃度分布
- CSVファイルとしてダウンロード可能

### 2. 🧮 拡散方程式ソルバー

**Fick's Second Law（Fickの第二法則）**を数値的に解きます：

```
∂C/∂t = ∂/∂x(D ∂C/∂x)
```

#### 実装された機能：

**拡散係数タイプ：**
- 定数拡散係数
- 濃度依存（線形）: D(C) = D₀ + (D₁ - D₀)C
- 濃度依存（非線形）: D(C) = D₀ + (D_max - D₀)C(1-C)

**初期条件：**
- ステップ関数（拡散対実験を模擬）
- 線形勾配
- ガウス分布

**数値解法：**
- 有限差分法（陽的オイラー法）
- 空間・時間分割数の調整可能
- 安定性チェック機能

### 3. 📈 高度な可視化

#### 3D表面プロット
- 時間と空間における濃度分布の3D可視化
- インタラクティブな回転・ズーム機能

#### アニメーション
- 時間発展のアニメーション表示
- フレーム数の調整可能
- 再生・停止コントロール

#### フラックス解析
- 拡散フラックス J = -D ∂C/∂x の計算と可視化
- 濃度勾配との関係表示

#### 比較プロット
- 計算結果と論文の実験データの比較
- 検証とバリデーション

### 4. 📚 理論背景

- CALPHADアプローチの説明
- 原子移動度の理論
- 活性化エネルギーの式
- 実験条件の詳細
- 物理定数の一覧

## 使用方法

### インストール

```bash
cd /home/ubuntu/repos/machine-learning
pip install -r requirements.txt
```

### アプリケーションの起動

```bash
streamlit run co_diffusion_app.py
```

ブラウザで自動的に開きます（通常は http://localhost:8501）

### 基本的な使い方

1. **論文データ抽出タブ**
   - Figure 19とFigure 20のデータを確認
   - CSVファイルをダウンロード

2. **拡散方程式ソルバータブ**
   - 左側のパラメータを設定：
     - 領域長さ、空間分割数
     - 拡散係数タイプと値
     - 初期条件
   - 「🚀 計算実行」ボタンをクリック
   - 結果を確認（濃度分布と時空間発展）
   - 時間スライダーで任意の時刻を表示
   - 結果をCSVでダウンロード

3. **可視化タブ**
   - 4つの可視化タイプから選択：
     - 3D表面プロット
     - アニメーション
     - フラックス解析
     - 比較プロット
   - インタラクティブに結果を探索

4. **理論背景タブ**
   - CALPHADアプローチの理論
   - 数学的定式化
   - 物理定数

## 技術仕様

### 数値解法

- **手法**: 有限差分法（陽的オイラー法）
- **境界条件**: Neumann境界条件（ゼロフラックス）
- **安定性**: 自動チェック（推奨: D·Δt/Δx² < 0.5）

### 拡散係数モデル

1. **定数**: D = D₀
2. **線形**: D(C) = D₀ + (D₁ - D₀)C
3. **非線形**: D(C) = D₀ + (D_max - D₀)C(1-C)

### 論文データ

**出典**: Lindwall, G., Moon, K., Williams, M., Tso, W., Campbell, C. (2024)
"Development of a Diffusion Mobility Database for Co-based Superalloys"
Journal of Phase Equilibria and Diffusion

**実験条件**:
- 温度: 1100°C (1373 K)
- 実験時間: 48h, 72h
- 合金系: Co-Al-Cr-Ni-Ti
- 手法: 拡散対実験

## パラメータ推奨値

### 典型的な拡散係数（1100°C）

- Co自己拡散: ~10-20 μm²/h
- Ni自己拡散: ~15-25 μm²/h
- Cr拡散: ~5-15 μm²/h

### 計算設定

- 領域長さ: 600 μm（拡散対実験に対応）
- 空間分割数: 200-500（精度と計算時間のバランス）
- 時間ステップ数: 500-1000
- 最終時間: 48-72 h

## 出力ファイル

### CSVファイル形式

**論文データ**:
```csv
Distance (μm),Co,Ni,Cr,Al,Ti
-300,0.7,0.0,0.25,0.05,0.0
...
```

**計算結果**:
```csv
Position (μm),t=0.00h,t=0.14h,...,t=72.00h
-300,0.7,0.699,...,0.45
...
```

## トラブルシューティング

### 計算が不安定

- 空間分割数を増やす（nx を大きく）
- 時間ステップ数を増やす（nt を大きく）
- 拡散係数を小さくする

### 計算が遅い

- 空間分割数を減らす
- 時間ステップ数を減らす
- 領域長さを短くする

### メモリエラー

- 分割数を減らす
- 最終時間を短くする

## 今後の拡張

- [ ] Crank-Nicolson法の実装（より高精度）
- [ ] 多成分系の同時拡散
- [ ] 温度依存性の考慮
- [ ] 実験データのアップロード機能
- [ ] パラメータフィッティング機能

## 参考文献

1. Lindwall, G., et al. (2024). "Development of a Diffusion Mobility Database for Co-based Superalloys", JPED.

2. Sato, J., et al. (2006). "Cobalt-Base High-Temperature Alloys", Science, 312(5770), 90-91.

3. Lass, E. A. (2017). "Application of computational thermodynamics to the design of a Co-Ni-based γ-γ′ superalloy", Metallurgical and Materials Transactions A, 48(5), 2443-2459.

4. Naghavi, S. S., et al. (2017). "Diffusivities and atomic mobilities in FCC Co-X (X= Ag, Au, Cu, Pd and Pt) alloys", Acta Materialia, 132, 467-478.

## ライセンス

このアプリケーションは教育・研究目的で作成されています。

## 作成者

Devin AI - 2024年10月24日

## バージョン

v1.0 - 初版リリース
