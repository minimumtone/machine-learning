#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
マテリアルズ・インフォマティクス教科書 - セットアップスクリプト
Materials Informatics Textbook - Setup Script
=============================================================================

【使用方法】
    # 開発モードでインストール（推奨）
    pip install -e .

    # 通常インストール
    pip install .

    # 依存パッケージのみインストール
    pip install -r requirements.txt

【対象】
    材料工学部 3回生

=============================================================================
"""

from setuptools import find_packages, setup

# パッケージ情報
PACKAGE_NAME = "mi_textbook"
VERSION = "1.0.0"
DESCRIPTION = "マテリアルズ・インフォマティクス教科書 コード集"
AUTHOR = "島根大学材料エネルギー学部"
PYTHON_REQUIRES = ">=3.8"

# 必須依存パッケージ
INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "scikit-learn>=1.0.0",
]

# オプション依存パッケージ
EXTRAS_REQUIRE = {
    "materials": [
        "pymatgen>=2022.0.0",
        "matminer>=0.7.0",
        "mp-api>=0.27.0",
    ],
    "deep_learning": [
        "torch>=1.9.0",
    ],
    "visualization": [
        "seaborn>=0.11.0",
    ],
    "dev": [
        "flake8>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "pytest>=6.0.0",
    ],
}

# 全てのオプション依存パッケージ
EXTRAS_REQUIRE["all"] = list(set(
    pkg for extras in EXTRAS_REQUIRE.values() for pkg in extras
))

# 長い説明文
LONG_DESCRIPTION = """
# マテリアルズ・インフォマティクス教科書 コード集

本パッケージは、島根大学材料エネルギー学部向け講義
「マテリアルズ・インフォマティックス応用」の教科書に掲載された
Pythonコードを整理・統一したものです。

## 対象
材料工学部 3回生

## モジュール一覧
- `materials_project_api`: Materials Project APIデータ取得
- `data_preprocessing`: データ前処理・クレンジング
- `pca_analysis`: 主成分分析（PCA）
- `regression_models`: 回帰モデル（線形回帰、Ridge、Lasso、ランダムフォレストなど）
- `classification_models`: 分類モデル（ロジスティック回帰、SVM、決定木など）
- `clustering_analysis`: クラスタリング分析（K-means、階層的、DBSCAN）
- `cross_validation`: 交差検証・ハイパーパラメータチューニング
- `bayesian_optimization`: ベイズ最適化・ガウス過程回帰
- `deep_learning_models`: 深層学習モデル
- `pymatgen_utils`: pymatgen構造操作ユーティリティ
- `matminer_features`: matminer特徴量生成

## インストール

```bash
# 基本インストール
pip install .

# 材料科学パッケージ込み
pip install .[materials]

# 全てのオプション込み
pip install .[all]
```

## 使用例

```python
from mi_textbook import regression_models

# 線形回帰モデルの学習
result = regression_models.train_linear_regression(x_train, y_train, x_test, y_test)
print(f"R²スコア: {result['test_metrics']['r2']:.4f}")
```
"""

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "materials informatics",
        "machine learning",
        "materials science",
        "pymatgen",
        "matminer",
    ],
)
