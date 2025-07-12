# 統計学習解析プラットフォーム

「統計学習入門 Python編」の解説事例をプログラム化した統合解析プラットフォームです。

## 概要

このプロジェクトは、統計学習の重要な概念と手法を実際のデータで学習できるStreamlitアプリケーション群です。各章の重要な例題とケーススタディを実装しています。

## 解析プログラム一覧

### 1. Boston住宅価格分析 (`boston_housing_analysis.py`)
- **概要**: 線形回帰による住宅価格予測
- **学習内容**: 単回帰、重回帰、モデル評価、残差分析
- **データセット**: Boston Housing Dataset (506サンプル, 13特徴量)

### 2. 自動車燃費分析 (`auto_mpg_analysis.py`)
- **概要**: 多項式回帰による非線形関係の分析
- **学習内容**: 多項式特徴量、非線形変換、モデル比較
- **データセット**: Auto Dataset (392サンプル, mpg vs horsepower)

### 3. 広告売上分析 (`advertising_analysis.py`)
- **概要**: 重回帰と交互作用効果の分析
- **学習内容**: 重回帰、交互作用、変数選択、予測精度
- **データセット**: Advertising Dataset (TV, Radio, Newspaper vs Sales)

### 4. 交差検証・ブートストラップ (`cross_validation_analysis.py`)
- **概要**: モデル選択と性能評価手法
- **学習内容**: 交差検証、ブートストラップ、バイアス-バリアンス、モデル選択
- **データセット**: 複数データセットでの比較分析

### 5. 分類分析 (`classification_analysis.py`)
- **概要**: ロジスティック回帰・LDA・QDA・KNN
- **学習内容**: ロジスティック回帰、線形判別分析、k近傍法、ROC曲線
- **データセット**: Stock Market Dataset, Iris Dataset

### 6. 決定木・アンサンブル (`tree_methods_analysis.py`)
- **概要**: 決定木・ランダムフォレスト・ブースティング
- **学習内容**: 決定木、ランダムフォレスト、ブースティング、特徴量重要度
- **データセット**: Boston Dataset, Heart Disease Dataset

## 使用方法

### 統合管理アプリケーション
```bash
streamlit run statistical_analysis_manager.py
```

### 個別プログラム実行
```bash
# 例: Boston住宅価格分析
streamlit run boston_housing_analysis.py
```

## インストール

必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

## 推奨学習順序

1. **Boston住宅価格分析** - 線形回帰の基礎
2. **自動車燃費分析** - 非線形関係の理解
3. **広告売上分析** - 重回帰と交互作用
4. **交差検証・ブートストラップ** - モデル評価
5. **分類分析** - 分類手法の比較
6. **決定木・アンサンブル** - 高度な手法

## テスト

テストスイートの実行:
```bash
python test_statistical_programs.py
```

## 特徴

- **インタラクティブな学習**: Streamlitによる直感的なUI
- **実データでの学習**: 実際のデータセットを使用した分析
- **視覚的な理解**: 豊富なグラフと可視化
- **パラメータ調整**: リアルタイムでのパラメータ変更
- **比較分析**: 複数手法の性能比較
- **教育的説明**: 各手法の理論的背景

## ファイル構成

```
machine-learning/
├── statistical_analysis_manager.py    # 統合管理アプリ
├── boston_housing_analysis.py         # Boston住宅価格分析
├── auto_mpg_analysis.py              # 自動車燃費分析
├── advertising_analysis.py           # 広告売上分析
├── cross_validation_analysis.py      # 交差検証・ブートストラップ
├── classification_analysis.py        # 分類分析
├── tree_methods_analysis.py          # 決定木・アンサンブル
├── test_statistical_programs.py      # テストスイート
├── requirements.txt                  # 依存パッケージ
└── README.md                         # このファイル
```

## 技術スタック

- **Python 3.12+**
- **Streamlit** - Webアプリケーションフレームワーク
- **scikit-learn** - 機械学習ライブラリ
- **pandas** - データ操作
- **numpy** - 数値計算
- **matplotlib/seaborn** - データ可視化
- **pytest** - テストフレームワーク

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
