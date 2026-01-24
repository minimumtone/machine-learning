#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
クラスタリング分析モジュール
=============================================================================

【学習目標】
    - クラスタリング（教師なし学習）の基本概念を理解する
    - 各種クラスタリング手法の特徴と使い分けを学ぶ
    - クラスタリング結果の評価方法を習得する

【前提知識】
    - 距離の概念（ユークリッド距離など）
    - 統計学の基礎
    - Python/NumPyの基本操作

【対象】
    材料工学部 3回生

【クラスタリングとは】
    クラスタリングは、ラベルのないデータを類似性に基づいて
    グループ（クラスタ）に分割する手法です。教師なし学習の代表例です。

【材料工学での応用例】
    - 類似材料のグループ化
    - 材料の分類体系の発見
    - 異常材料（外れ値）の検出

=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# クラスタリング評価指標
# =============================================================================

def calculate_clustering_metrics(
    data: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    クラスタリング結果の評価指標を計算する。

    【主な評価指標】
    - シルエットスコア: -1〜1の範囲。1に近いほど良いクラスタリング
    - Calinski-Harabaszスコア: 大きいほど良い。クラスタ間分散/クラスタ内分散
    - Davies-Bouldinスコア: 小さいほど良い。クラスタ間の類似度

    Args:
        data: 入力データ
        labels: クラスタラベル

    Returns:
        評価指標を含む辞書
    """
    # ラベルが1種類のみの場合は評価不可
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None
        }

    # ノイズラベル（-1）を除外
    mask = labels != -1
    if mask.sum() < 2:
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None
        }

    data_filtered = data[mask]
    labels_filtered = labels[mask]

    if len(np.unique(labels_filtered)) < 2:
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None
        }

    return {
        'silhouette': silhouette_score(data_filtered, labels_filtered),
        'calinski_harabasz': calinski_harabasz_score(data_filtered, labels_filtered),
        'davies_bouldin': davies_bouldin_score(data_filtered, labels_filtered)
    }


# =============================================================================
# K-meansクラスタリング
# =============================================================================

def train_kmeans(
    data: np.ndarray,
    n_clusters: int = 3,
    standardize: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    K-meansクラスタリングを実行する。

    K-meansは、データをk個のクラスタに分割する最も基本的な手法です。
    各クラスタの重心を反復的に更新して、クラスタ内の分散を最小化します。

    【アルゴリズム】
    1. k個の初期重心をランダムに選択
    2. 各データ点を最も近い重心のクラスタに割り当て
    3. 各クラスタの重心を再計算
    4. 収束するまで2-3を繰り返し

    【注意点】
    - クラスタ数kを事前に指定する必要がある
    - 初期値に依存する（複数回実行を推奨）
    - 球状のクラスタを仮定

    Args:
        data: 入力データ
        n_clusters: クラスタ数
        standardize: 標準化を行うかどうか
        random_state: 乱数シード

    Returns:
        クラスタリング結果を含む辞書
    """
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        scaler = None
        data_scaled = data

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(data_scaled)

    return {
        'model': model,
        'labels': labels,
        'centroids': model.cluster_centers_,
        'inertia': model.inertia_,
        'scaler': scaler,
        'metrics': calculate_clustering_metrics(data_scaled, labels)
    }


def find_optimal_k(
    data: np.ndarray,
    k_range: range = range(2, 11),
    standardize: bool = True
) -> Dict[str, Any]:
    """
    エルボー法とシルエット法で最適なクラスタ数を探索する。

    【エルボー法】
    クラスタ数を増やしていくと、クラスタ内分散（inertia）は減少します。
    減少率が急激に緩やかになる点（エルボー）が最適なk。

    【シルエット法】
    シルエットスコアが最大となるkを選択。

    Args:
        data: 入力データ
        k_range: 探索するクラスタ数の範囲
        standardize: 標準化を行うかどうか

    Returns:
        各kでの評価結果を含む辞書
    """
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data

    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data_scaled, labels))

    # 最適なkを決定（シルエットスコア最大）
    optimal_k = k_range[np.argmax(silhouettes)]

    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'optimal_k': optimal_k
    }


# =============================================================================
# 階層的クラスタリング
# =============================================================================

def train_hierarchical(
    data: np.ndarray,
    n_clusters: int = 3,
    linkage: str = 'ward',
    standardize: bool = True
) -> Dict[str, Any]:
    """
    階層的クラスタリングを実行する。

    階層的クラスタリングは、データを階層的な木構造（デンドログラム）で
    表現します。凝集型（ボトムアップ）と分割型（トップダウン）があります。

    【リンケージ法】
    - 'ward': ウォード法（クラスタ内分散を最小化）- 最も一般的
    - 'complete': 完全連結法（最大距離）
    - 'average': 平均連結法（平均距離）
    - 'single': 単連結法（最小距離）

    Args:
        data: 入力データ
        n_clusters: クラスタ数
        linkage: リンケージ法
        standardize: 標準化を行うかどうか

    Returns:
        クラスタリング結果を含む辞書
    """
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        scaler = None
        data_scaled = data

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(data_scaled)

    return {
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'linkage': linkage,
        'scaler': scaler,
        'metrics': calculate_clustering_metrics(data_scaled, labels)
    }


# =============================================================================
# DBSCANクラスタリング
# =============================================================================

def train_dbscan(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    DBSCANクラスタリングを実行する。

    DBSCAN（Density-Based Spatial Clustering of Applications with Noise）は、
    密度に基づくクラスタリング手法です。任意の形状のクラスタを検出でき、
    外れ値（ノイズ）を自動的に識別します。

    【パラメータ】
    - eps: 近傍の半径。この距離内の点を近傍とみなす
    - min_samples: コアポイントとなるための最小近傍点数

    【特徴】
    - クラスタ数を事前に指定不要
    - 任意形状のクラスタを検出可能
    - 外れ値を自動検出（ラベル=-1）

    Args:
        data: 入力データ
        eps: 近傍の半径
        min_samples: 最小サンプル数
        standardize: 標準化を行うかどうか

    Returns:
        クラスタリング結果を含む辞書
    """
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        scaler = None
        data_scaled = data

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data_scaled)

    # クラスタ数とノイズ点数を計算
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return {
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'min_samples': min_samples,
        'scaler': scaler,
        'metrics': calculate_clustering_metrics(data_scaled, labels)
    }


# =============================================================================
# 可視化関数
# =============================================================================

def plot_clusters_2d(
    data: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: str = "Clustering Result",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    クラスタリング結果を2次元で可視化する。

    Args:
        data: 入力データ（2次元）
        labels: クラスタラベル
        centroids: クラスタ重心（オプション）
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ユニークなラベル
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # ノイズ点は灰色で表示
            color = 'gray'
            marker = 'x'
            label_name = 'Noise'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'

        mask = labels == label
        ax.scatter(
            data[mask, 0], data[mask, 1],
            c=[color], marker=marker, label=label_name,
            alpha=0.7, edgecolors='white', linewidth=0.5
        )

    # 重心をプロット
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c='red', marker='*', s=200, label='Centroids',
            edgecolors='black', linewidth=1
        )

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_elbow_silhouette(
    k_results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    エルボー法とシルエット法の結果を可視化する。

    Args:
        k_results: find_optimal_kの結果
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    k_range = k_results['k_range']

    # エルボー法
    axes[0].plot(k_range, k_results['inertias'], 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)

    # シルエット法
    axes[1].plot(k_range, k_results['silhouettes'], 'go-')
    axes[1].axvline(
        x=k_results['optimal_k'],
        color='r', linestyle='--',
        label=f"Optimal k = {k_results['optimal_k']}"
    )
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Method')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_silhouette_analysis(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Silhouette Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    シルエット分析を可視化する。

    各サンプルのシルエット係数を可視化し、クラスタの品質を評価します。
    シルエット係数が高いサンプルは、適切なクラスタに属しています。

    Args:
        data: 入力データ
        labels: クラスタラベル
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    # シルエット係数を計算
    silhouette_vals = silhouette_samples(data, labels)
    silhouette_avg = silhouette_score(data, labels)

    y_lower = 10
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    for i, label in enumerate(unique_labels):
        cluster_silhouette_vals = silhouette_vals[labels == label]
        cluster_silhouette_vals.sort()

        cluster_size = len(cluster_silhouette_vals)
        y_upper = y_lower + cluster_size

        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette_vals,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(label))

        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color='red', linestyle='--',
               label=f'Average: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("クラスタリング分析モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    from sklearn.datasets import make_blobs

    x_data, y_true = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.6,
        random_state=42
    )

    print(f"データ形状: {x_data.shape}")
    print(f"真のクラスタ数: {len(np.unique(y_true))}")

    # 最適なクラスタ数の探索
    print("\n【2. 最適なクラスタ数の探索】")
    print("-" * 50)

    k_results = find_optimal_k(x_data, k_range=range(2, 10))
    print(f"最適なクラスタ数: {k_results['optimal_k']}")

    # K-meansクラスタリング
    print("\n【3. K-meansクラスタリング】")
    print("-" * 50)

    kmeans_result = train_kmeans(x_data, n_clusters=4)
    print(f"クラスタ数: 4")
    print(f"シルエットスコア: {kmeans_result['metrics']['silhouette']:.4f}")

    # 階層的クラスタリング
    print("\n【4. 階層的クラスタリング】")
    print("-" * 50)

    hierarchical_result = train_hierarchical(x_data, n_clusters=4)
    print(f"クラスタ数: 4")
    print(f"シルエットスコア: {hierarchical_result['metrics']['silhouette']:.4f}")

    # DBSCANクラスタリング
    print("\n【5. DBSCANクラスタリング】")
    print("-" * 50)

    dbscan_result = train_dbscan(x_data, eps=0.5, min_samples=5)
    print(f"検出クラスタ数: {dbscan_result['n_clusters']}")
    print(f"ノイズ点数: {dbscan_result['n_noise']}")

    # 可視化
    print("\n【6. 可視化】")
    print("-" * 50)

    fig1 = plot_clusters_2d(
        x_data, kmeans_result['labels'],
        centroids=kmeans_result['centroids'],
        title="K-means Clustering Result"
    )
    plt.close(fig1)
    print("K-meansクラスタリング結果: 作成完了")

    fig2 = plot_elbow_silhouette(k_results)
    plt.close(fig2)
    print("エルボー・シルエット分析: 作成完了")

    print("\n処理完了!")
