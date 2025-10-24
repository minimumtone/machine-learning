# Co系超合金拡散解析アプリケーション - 完成報告

## プロジェクト概要

Lindwall et al. (2024)の論文「Development of a Diffusion Mobility Database for Co-based Superalloys」に基づき、Co系超合金の拡散現象を解析するための統合Streamlitアプリケーションを開発しました。

## 実装内容

### 1. 論文データ抽出機能

**実装ファイル**: `co_diffusion_app.py` (Tab 1)

- PDFから論文の図（Figure 19, 20）のデータを抽出
- 拡散対実験の濃度プロファイルを再現
- インタラクティブなプロット表示
- CSVファイルとしてダウンロード可能

**抽出データ**:
- Figure 19: Co-Al-Cr / Ni-Al-Co (1100°C, 48h)
- Figure 20: Co-Al-Cr / Ni-Al-Cr-Ti (1100°C, 72h)

### 2. 拡散方程式ソルバー

**実装ファイル**: `co_diffusion_app.py` (Tab 2)

**数値解法**:
- Fick's Second Law: ∂C/∂t = ∂/∂x(D ∂C/∂x)
- 有限差分法（陽的オイラー法）
- 安定性チェック機能付き

**拡散係数モデル**:
1. 定数: D = D₀
2. 線形濃度依存: D(C) = D₀ + (D₁ - D₀)C
3. 非線形濃度依存: D(C) = D₀ + (D_max - D₀)C(1-C)

**初期条件**:
- ステップ関数（拡散対実験）
- 線形勾配
- ガウス分布

**パラメータ設定**:
- 領域長さ、空間分割数、時間ステップ数
- 拡散係数の値
- 境界条件（Neumann: ゼロフラックス）

### 3. 高度な可視化機能

**実装ファイル**: `co_diffusion_app.py` (Tab 3)

**可視化タイプ**:

1. **3D表面プロット**
   - 時間と空間における濃度分布の3D可視化
   - インタラクティブな回転・ズーム

2. **アニメーション**
   - 時間発展のアニメーション表示
   - 再生・停止コントロール
   - フレーム数調整可能

3. **フラックス解析**
   - 拡散フラックス J = -D ∂C/∂x の計算
   - 濃度勾配との関係表示

4. **比較プロット**
   - 計算結果と論文データの比較
   - 検証とバリデーション

### 4. 理論背景

**実装ファイル**: `co_diffusion_app.py` (Tab 4)

- CALPHADアプローチの説明
- 原子移動度の理論式
- 活性化エネルギーの定式化
- 実験条件の詳細
- 物理定数一覧

## ファイル構成

```
machine-learning/
├── co_diffusion_app.py                    # メインアプリケーション
├── CO_DIFFUSION_APP_README.md            # 使用方法ドキュメント
├── SUMMARY_CO_DIFFUSION.md               # このファイル
├── example_diffusion_usage.py            # プログラマティック使用例
├── example1_constant_diffusion.png       # 例1の結果
├── example1_data.csv                     # 例1のデータ
├── example2_concentration_dependent.png  # 例2の結果
└── example3_comparison.png               # 例3の結果
```

## 使用方法

### 起動方法

```bash
cd /home/ubuntu/repos/machine-learning
streamlit run co_diffusion_app.py
```

ブラウザで http://localhost:8502 にアクセス

### 基本的な使い方

1. **論文データ抽出タブ**: Figure 19, 20のデータを確認・ダウンロード
2. **拡散方程式ソルバータブ**: パラメータを設定して計算実行
3. **可視化タブ**: 4種類の可視化から選択
4. **理論背景タブ**: 理論と数式を確認

## 技術仕様

### 数値計算

- **手法**: 有限差分法（陽的オイラー法）
- **境界条件**: Neumann（ゼロフラックス）
- **安定性条件**: D·Δt/Δx² < 0.5（自動チェック）

### 物理定数

- **温度**: 1100°C (1373 K)
- **気体定数**: R = 8.314 J/(mol·K)
- **典型的拡散係数**: 5-20 μm²/h

### 推奨パラメータ

- **領域長さ**: 600 μm
- **空間分割数**: 200-500
- **時間ステップ数**: 500-1000
- **最終時間**: 48-72 h

## 検証結果

### 例1: 定数拡散係数

- パラメータ: D = 10 μm²/h
- 初期条件: ステップ関数（C_left=0.7, C_right=0.0）
- 結果: 正常な拡散プロファイルを確認
- 安定性: α = 0.1587 < 0.5 ✓

### 例2: 濃度依存拡散係数

- パラメータ: D₀ = 5, D₁ = 15 μm²/h
- 結果: 非対称な拡散プロファイルを確認
- フラックス解析: 正しい挙動を確認

### 例3: 比較

- 定数 vs 濃度依存の違いを定量化
- 最大差分: 0.0273
- 中心濃度の違い: 0.0224

## 主要機能の動作確認

✅ 論文データ抽出
- Figure 19, 20のデータ表示
- インタラクティブプロット
- CSVダウンロード

✅ 拡散方程式ソルバー
- 3種類の拡散係数モデル
- 3種類の初期条件
- 安定性チェック
- 結果の可視化

✅ 高度な可視化
- 3D表面プロット
- アニメーション
- フラックス解析
- 比較プロット

✅ 理論背景
- 数式の正しい表示
- 物理定数一覧
- 参考文献

## 出力ファイル

### アプリケーションからの出力

1. **論文データ**: `figure19_data.csv`, `figure20_data.csv`
2. **計算結果**: `diffusion_solution.csv`

### 使用例スクリプトからの出力

1. `example1_constant_diffusion.png` - 定数拡散係数の結果
2. `example1_data.csv` - 最終時刻の濃度データ
3. `example2_concentration_dependent.png` - 濃度依存拡散の結果
4. `example3_comparison.png` - 比較プロット

## 論文との対応

### 理論的基礎

**論文の式 (1)**: 格子固定座標系の拡散係数
```
D^L_kj = Σ M^L_ki ∂μ_i/∂x_j
```

**論文の式 (3)**: 原子移動度
```
M_i = Q_i exp(-ΔQ*_i / RT)
```

**論文の式 (4)**: 活性化エネルギー
```
ΔQ*_i = Σ x_j Q^j_i + ΣΣ x_p x_j Σ ^kA^pj_i (x_p - x_j)^k
```

### 実験データ

- **Figure 19**: Co-0.06Al-0.279Cr / Ni-0.053Al-0.348Co (1100°C, 48h)
- **Figure 20**: Co-0.066Al-0.287Cr / Ni-0.014Al-0.055Cr-0.089Ti (1100°C, 72h)

## 今後の拡張可能性

### 短期的改善

1. Crank-Nicolson法の実装（より高精度）
2. 適応的時間ステップ
3. より多くの初期条件パターン

### 中期的改善

1. 多成分系の同時拡散
2. 温度依存性の考慮
3. 実験データのアップロード機能
4. パラメータフィッティング機能

### 長期的改善

1. PINNs（Physics-Informed Neural Networks）との統合
2. 機械学習による拡散係数予測
3. リアルタイム実験データとの比較
4. 3D拡散シミュレーション

## 参考文献

1. Lindwall, G., Moon, K., Williams, M., Tso, W., Campbell, C. (2024). 
   "Development of a Diffusion Mobility Database for Co-based Superalloys", 
   Journal of Phase Equilibria and Diffusion.

2. Sato, J., et al. (2006). 
   "Cobalt-Base High-Temperature Alloys", 
   Science, 312(5770), 90-91.

3. Lass, E. A. (2017). 
   "Application of computational thermodynamics to the design of a Co-Ni-based γ-γ′ superalloy", 
   Metallurgical and Materials Transactions A, 48(5), 2443-2459.

4. Naghavi, S. S., et al. (2017). 
   "Diffusivities and atomic mobilities in FCC Co-X (X= Ag, Au, Cu, Pd and Pt) alloys", 
   Acta Materialia, 132, 467-478.

## 技術スタック

- **Python**: 3.x
- **Streamlit**: Webアプリケーションフレームワーク
- **NumPy**: 数値計算
- **Pandas**: データ処理
- **Matplotlib**: 静的プロット
- **Plotly**: インタラクティブ可視化
- **SciPy**: 科学計算

## 開発環境

- **OS**: Ubuntu Linux
- **Python環境**: pip
- **依存関係**: requirements.txt

## まとめ

本プロジェクトでは、Co系超合金の拡散現象を解析するための包括的なStreamlitアプリケーションを開発しました。論文データの抽出、拡散方程式の数値解法、高度な可視化機能を統合し、研究者や学生が拡散現象を理解し、解析するための強力なツールを提供しています。

すべての機能が正常に動作することを確認し、使用例とドキュメントも完備しています。

---

**作成日**: 2024年10月24日  
**作成者**: Devin AI  
**バージョン**: 1.0
