#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
主成分分析（PCA）モジュール
=============================================================================

【学習目標】
    - 主成分分析（PCA）の原理と目的を理解する
    - 次元削減の重要性と応用場面を学ぶ
    - PCAの結果を可視化・解釈する方法を習得する

【前提知識】
    - 線形代数の基礎（固有値、固有ベクトル）
    - 分散・共分散の概念
    - NumPy/Pandasの基本操作

【対象】
    材料工学部 3回生

【PCAとは】
    主成分分析（Principal Component Analysis）は、高次元データを
    低次元に圧縮する手法です。データの分散が最大となる方向を
    順番に見つけ、それを新しい座標軸（主成分）とします。

【材料工学での応用例】
    - 多数の物性値から材料の特徴を抽出
    - 類似材料のグループ化
    - 異常値（欠陥材料）の検出

=============================================================================
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PCA実行関数
# =============================================================================

def perform_pca(
    data: Union[np.ndarray, pd.DataFrame],
    n_components: Optional[int] = None,
    standardize: bool = True
) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]:
    """
    主成分分析を実行する。

    PCAは以下の手順で実行されます：
    1. データの標準化（オプション）
    2. 共分散行列の計算
    3. 固有値・固有ベクトルの計算
    4. 主成分への射影

    Args:
        data: 入力データ（サンプル数 x 特徴量数）
        n_components: 抽出する主成分数（Noneの場合は全て）
        standardize: 標準化を行うかどうか（推奨: True）

    Returns:
        変換後のデータ、PCAオブジェクト、スケーラー（使用時）のタプル

    Example:
        >>> data = np.random.randn(100, 5)
        >>> transformed, pca, scaler = perform_pca(data, n_components=2)
        >>> print(f"変換後の形状: {transformed.shape}")
    """
    # DataFrameの場合はNumPy配列に変換
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data

    scaler = None

    # 標準化（各特徴量を平均0、標準偏差1に変換）
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_array)
    else:
        data_scaled = data_array

    # PCAの実行
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data_scaled)

    return transformed_data, pca, scaler


# =============================================================================
# 可視化関数
# =============================================================================

def plot_pca_variance(
    pca: PCA,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    主成分の寄与率を可視化する。

    寄与率（explained variance ratio）は、各主成分がデータの
    分散をどれだけ説明しているかを示します。累積寄与率が
    80-90%になる主成分数を選ぶことが一般的です。

    Args:
        pca: 学習済みPCAオブジェクト
        save_path: 保存先パス（Noneの場合は保存しない）
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n_components = len(pca.explained_variance_ratio_)
    x_values = range(1, n_components + 1)

    # 個別寄与率
    axes[0].bar(x_values, pca.explained_variance_ratio_, alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Explained Variance')
    axes[0].set_xticks(x_values)

    # 累積寄与率
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(x_values, cumulative_variance, 'bo-')
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    axes[1].axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].set_xticks(x_values)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pca_2d(
    transformed_data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "PCA Result (2D)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    PCA結果を2次元散布図で可視化する。

    2次元に削減したデータを散布図として表示します。
    ラベルがある場合は色分けして表示します。

    Args:
        transformed_data: PCA変換後のデータ
        labels: サンプルのラベル（クラスタリング結果など）
        feature_names: 元の特徴量名（バイプロット用）
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pca_loadings(
    pca: PCA,
    feature_names: List[str],
    n_components: int = 2,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    PCAの因子負荷量（ローディング）を可視化する。

    因子負荷量は、各主成分と元の特徴量との相関を示します。
    絶対値が大きい特徴量ほど、その主成分に強く寄与しています。

    【解釈のポイント】
    - 正の負荷量: 特徴量が増加すると主成分も増加
    - 負の負荷量: 特徴量が増加すると主成分は減少
    - 絶対値が大きい: その特徴量の影響が大きい

    Args:
        pca: 学習済みPCAオブジェクト
        feature_names: 特徴量名のリスト
        n_components: 表示する主成分数
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    loadings = pca.components_[:n_components].T
    n_features = len(feature_names)

    x_positions = np.arange(n_features)
    bar_width = 0.35

    for i in range(n_components):
        offset = (i - n_components / 2 + 0.5) * bar_width
        ax.bar(
            x_positions + offset,
            loadings[:, i],
            bar_width,
            label=f'PC{i + 1}',
            alpha=0.7
        )

    ax.set_xlabel('Features')
    ax.set_ylabel('Loading')
    ax.set_title('PCA Loadings')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# 分析関数
# =============================================================================

def analyze_pca_results(
    pca: PCA,
    feature_names: Optional[List[str]] = None,
    variance_threshold: float = 0.9
) -> dict:
    """
    PCA結果を分析してサマリーを返す。

    Args:
        pca: 学習済みPCAオブジェクト
        feature_names: 特徴量名のリスト
        variance_threshold: 累積寄与率の閾値

    Returns:
        分析結果を含む辞書
    """
    results = {}

    # 寄与率
    results['explained_variance_ratio'] = pca.explained_variance_ratio_
    results['cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)

    # 閾値を超える主成分数
    n_components_threshold = np.argmax(
        results['cumulative_variance'] >= variance_threshold
    ) + 1
    results['n_components_for_threshold'] = n_components_threshold
    results['variance_threshold'] = variance_threshold

    # 各主成分の主要な特徴量
    if feature_names is not None:
        top_features = {}
        for i, component in enumerate(pca.components_):
            sorted_indices = np.argsort(np.abs(component))[::-1]
            top_features[f'PC{i + 1}'] = [
                (feature_names[idx], component[idx])
                for idx in sorted_indices[:3]
            ]
        results['top_features_per_component'] = top_features

    return results


def select_n_components(
    data: Union[np.ndarray, pd.DataFrame],
    variance_threshold: float = 0.9,
    standardize: bool = True
) -> int:
    """
    指定した累積寄与率を達成する主成分数を決定する。

    Args:
        data: 入力データ
        variance_threshold: 目標とする累積寄与率
        standardize: 標準化を行うかどうか

    Returns:
        必要な主成分数

    Example:
        >>> n = select_n_components(data, variance_threshold=0.95)
        >>> print(f"95%の分散を説明するには{n}個の主成分が必要")
    """
    _, pca, _ = perform_pca(data, n_components=None, standardize=standardize)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    return n_components


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("主成分分析（PCA）モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成（材料物性を模擬）
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    np.random.seed(42)
    n_samples = 100

    # 相関のある特徴量を生成
    base = np.random.randn(n_samples)
    sample_data = pd.DataFrame({
        'density': 2.5 + 0.3 * base + 0.1 * np.random.randn(n_samples),
        'bandgap': 1.5 + 0.2 * base + 0.1 * np.random.randn(n_samples),
        'formation_energy': -0.5 - 0.1 * base + 0.05 * np.random.randn(n_samples),
        'volume': 50 + 5 * base + 2 * np.random.randn(n_samples),
        'elastic_modulus': 100 + 10 * np.random.randn(n_samples)
    })

    print(f"データ形状: {sample_data.shape}")
    print(f"特徴量: {list(sample_data.columns)}")

    # PCAの実行
    print("\n【2. PCAの実行】")
    print("-" * 50)

    transformed, pca, scaler = perform_pca(sample_data, n_components=None)

    print("寄与率:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        cumulative = sum(pca.explained_variance_ratio_[:i + 1])
        print(f"  PC{i + 1}: {ratio:.4f} (累積: {cumulative:.4f})")

    # 分析結果
    print("\n【3. 分析結果】")
    print("-" * 50)

    analysis = analyze_pca_results(
        pca,
        feature_names=list(sample_data.columns),
        variance_threshold=0.9
    )

    print(f"90%の分散を説明する主成分数: {analysis['n_components_for_threshold']}")
    print("\n各主成分の主要な特徴量:")
    for pc, features in analysis['top_features_per_component'].items():
        print(f"  {pc}:")
        for name, loading in features:
            print(f"    {name}: {loading:.4f}")

    # 可視化
    print("\n【4. 可視化】")
    print("-" * 50)

    # 寄与率プロット
    fig1 = plot_pca_variance(pca)
    plt.close(fig1)
    print("寄与率プロット: 作成完了")

    # 2D散布図
    fig2 = plot_pca_2d(transformed, title="Material Properties PCA")
    plt.close(fig2)
    print("2D散布図: 作成完了")

    # ローディングプロット
    fig3 = plot_pca_loadings(pca, list(sample_data.columns))
    plt.close(fig3)
    print("ローディングプロット: 作成完了")

    print("\n処理完了!")
