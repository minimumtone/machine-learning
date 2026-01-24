#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
マテリアルズ・インフォマティクス教科書 コード集
Materials Informatics Textbook Code Collection
=============================================================================

【概要】
    本パッケージは、島根大学材料エネルギー学部向け講義
    「マテリアルズ・インフォマティックス応用」の教科書に掲載された
    Pythonコードを整理・統一したものです。

【対象】
    材料工学部 3回生

【モジュール一覧】
    - materials_project_api: Materials Project APIデータ取得
    - data_preprocessing: データ前処理・クレンジング
    - pca_analysis: 主成分分析（PCA）
    - regression_models: 回帰モデル各種
    - classification_models: 分類モデル各種
    - clustering_analysis: クラスタリング手法
    - cross_validation: 交差検証
    - bayesian_optimization: ベイズ最適化・ガウス過程回帰
    - deep_learning_models: 深層学習モデル
    - pymatgen_utils: pymatgen構造操作ユーティリティ
    - matminer_features: matminer特徴量生成

【使用方法】
    >>> from mi_textbook import regression_models
    >>> results = regression_models.train_linear_regression(x_train, y_train)

=============================================================================
"""

__version__ = "1.0.0"
__author__ = "Materials Informatics Textbook"
__all__ = [
    "materials_project_api",
    "data_preprocessing",
    "pca_analysis",
    "regression_models",
    "classification_models",
    "clustering_analysis",
    "cross_validation",
    "bayesian_optimization",
    "deep_learning_models",
    "pymatgen_utils",
    "matminer_features",
]
