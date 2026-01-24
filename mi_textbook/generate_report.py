#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
レポート生成スクリプト
各モジュールの実行結果と可視化を生成する
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 警告を抑制
warnings.filterwarnings('ignore')

# 出力ディレクトリ
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_pca_figures():
    """PCA分析の可視化を生成"""
    print("PCA分析の可視化を生成中...")
    from mi_textbook import pca_analysis as pca

    # サンプルデータ（材料物性を模擬）
    np.random.seed(42)
    n_samples = 100
    base = np.random.randn(n_samples)
    sample_data = pd.DataFrame({
        'density': 2.5 + 0.3 * base + 0.1 * np.random.randn(n_samples),
        'bandgap': 1.5 + 0.2 * base + 0.1 * np.random.randn(n_samples),
        'formation_energy': -0.5 - 0.1 * base + 0.05 * np.random.randn(n_samples),
        'volume': 50 + 5 * base + 2 * np.random.randn(n_samples),
        'elastic_modulus': 100 + 10 * np.random.randn(n_samples)
    })

    # PCA実行
    transformed, pca_obj, scaler = pca.perform_pca(sample_data, n_components=None)

    # 寄与率プロット
    fig = pca.plot_pca_variance(pca_obj)
    fig.savefig(os.path.join(FIGURES_DIR, 'pca_variance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2D散布図
    fig = pca.plot_pca_2d(transformed, title="Material Properties PCA")
    fig.savefig(os.path.join(FIGURES_DIR, 'pca_2d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ローディングプロット
    fig = pca.plot_pca_loadings(pca_obj, list(sample_data.columns))
    fig.savefig(os.path.join(FIGURES_DIR, 'pca_loadings.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  - pca_variance.png")
    print("  - pca_2d.png")
    print("  - pca_loadings.png")

    return {
        'explained_variance': pca_obj.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca_obj.explained_variance_ratio_)
    }


def generate_regression_figures():
    """回帰モデルの可視化を生成"""
    print("回帰モデルの可視化を生成中...")
    from mi_textbook import regression_models as reg
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # サンプルデータ
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 各モデルの学習と評価
    models_results = {}

    # 線形回帰
    result = reg.train_linear_regression(X_train, y_train, X_test, y_test)
    models_results['Linear'] = result['test_metrics']['r2']

    # Ridge回帰
    result = reg.train_ridge_regression(X_train, y_train, alpha=1.0, x_test=X_test, y_test=y_test)
    models_results['Ridge'] = result['test_metrics']['r2']

    # Lasso回帰
    result = reg.train_lasso_regression(X_train, y_train, alpha=0.1, x_test=X_test, y_test=y_test)
    models_results['Lasso'] = result['test_metrics']['r2']

    # ランダムフォレスト
    result = reg.train_random_forest_regressor(X_train, y_train, x_test=X_test, y_test=y_test)
    models_results['RandomForest'] = result['test_metrics']['r2']

    # 比較プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(models_results.keys())
    r2_scores = list(models_results.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, r2_scores, color=colors)
    ax.set_ylabel('R² Score')
    ax.set_title('Regression Models Comparison')
    ax.set_ylim(0, 1.1)
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'regression_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  - regression_comparison.png")

    return models_results


def generate_classification_figures():
    """分類モデルの可視化を生成"""
    print("分類モデルの可視化を生成中...")
    from mi_textbook import classification_models as clf
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # サンプルデータ
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2,
                               n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 各モデルの学習と評価
    models_results = {}

    # ロジスティック回帰
    result = clf.train_logistic_regression(X_train, y_train, c_param=1.0, x_test=X_test, y_test=y_test)
    models_results['Logistic'] = result['test_metrics']['accuracy']

    # SVM
    result = clf.train_svm_classifier(X_train, y_train, x_test=X_test, y_test=y_test)
    models_results['SVM'] = result['test_metrics']['accuracy']

    # k-NN
    result = clf.train_knn_classifier(X_train, y_train, x_test=X_test, y_test=y_test)
    models_results['k-NN'] = result['test_metrics']['accuracy']

    # ランダムフォレスト
    result = clf.train_random_forest_classifier(X_train, y_train, x_test=X_test, y_test=y_test)
    models_results['RandomForest'] = result['test_metrics']['accuracy']

    # 比較プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(models_results.keys())
    accuracies = list(models_results.values())
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, accuracies, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Models Comparison')
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'classification_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 混同行列
    result = clf.train_random_forest_classifier(X_train, y_train, x_test=X_test, y_test=y_test)
    fig = clf.plot_confusion_matrix(y_test, result['test_predictions'],
                                    title="Random Forest Confusion Matrix")
    fig.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  - classification_comparison.png")
    print("  - confusion_matrix.png")

    return models_results


def generate_clustering_figures():
    """クラスタリングの可視化を生成"""
    print("クラスタリングの可視化を生成中...")
    from mi_textbook import clustering_analysis as clust
    from sklearn.datasets import make_blobs

    # サンプルデータ
    X, _ = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)

    # 最適なk探索
    k_results = clust.find_optimal_k(X, k_range=range(2, 10))

    # エルボー法・シルエット法プロット
    fig = clust.plot_elbow_silhouette(k_results)
    fig.savefig(os.path.join(FIGURES_DIR, 'elbow_silhouette.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # K-meansクラスタリング
    result = clust.train_kmeans(X, n_clusters=4)
    fig = clust.plot_clusters_2d(X, result['labels'], result['centroids'],
                                 title="K-means Clustering (k=4)")
    fig.savefig(os.path.join(FIGURES_DIR, 'kmeans_clusters.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # シルエット分析
    fig = clust.plot_silhouette_analysis(X, result['labels'])
    fig.savefig(os.path.join(FIGURES_DIR, 'silhouette_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  - elbow_silhouette.png")
    print("  - kmeans_clusters.png")
    print("  - silhouette_analysis.png")

    return {
        'optimal_k': k_results['optimal_k'],
        'silhouette_score': result['metrics']['silhouette']
    }


def generate_cross_validation_figures():
    """交差検証の可視化を生成"""
    print("交差検証の可視化を生成中...")
    from mi_textbook import cross_validation as cv
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge

    # サンプルデータ
    X, y = make_regression(n_samples=200, n_features=10, noise=20, random_state=42)

    # 学習曲線
    model = Ridge(alpha=1.0)
    lc_results = cv.compute_learning_curve(model, X, y)
    fig = cv.plot_learning_curve(lc_results)
    fig.savefig(os.path.join(FIGURES_DIR, 'learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # グリッドサーチ
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    gs_results = cv.perform_grid_search(Ridge(), param_grid, X, y)
    fig = cv.plot_grid_search_results(gs_results, param_name='alpha')
    fig.savefig(os.path.join(FIGURES_DIR, 'grid_search.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  - learning_curve.png")
    print("  - grid_search.png")

    return {
        'best_params': gs_results['best_params'],
        'best_score': gs_results['best_score']
    }


def generate_bayesian_optimization_figures():
    """ベイズ最適化の可視化を生成"""
    print("ベイズ最適化の可視化を生成中...")
    from mi_textbook import bayesian_optimization as bo

    # サンプルデータ（1次元関数）
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, 10).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + np.random.normal(0, 0.1, 10)

    # GPR学習
    result = bo.train_gpr(X_train, y_train)

    # GPR予測の可視化
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred, y_std, ci_lower, ci_upper = bo.predict_with_uncertainty(result['model'], X_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train, y_train, c='red', s=50, zorder=10, label='Training data')
    ax.plot(X_test, y_pred, 'b-', label='GPR prediction')
    ax.fill_between(X_test.flatten(),
                    y_pred - 2*y_std, y_pred + 2*y_std,
                    alpha=0.3, color='blue', label='95% confidence')
    ax.plot(X_test, np.sin(X_test), 'g--', label='True function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gaussian Process Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'gpr_prediction.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  - gpr_prediction.png")

    return {'r2_score': result.get('r2_score', 'N/A')}


def main():
    """メイン関数"""
    print("=" * 60)
    print("マテリアルズ・インフォマティクス教科書")
    print("実行結果レポート生成")
    print("=" * 60)

    results = {}

    # 各モジュールの可視化を生成
    results['pca'] = generate_pca_figures()
    results['regression'] = generate_regression_figures()
    results['classification'] = generate_classification_figures()
    results['clustering'] = generate_clustering_figures()
    results['cross_validation'] = generate_cross_validation_figures()
    results['bayesian'] = generate_bayesian_optimization_figures()

    print("\n" + "=" * 60)
    print("可視化生成完了!")
    print(f"出力ディレクトリ: {FIGURES_DIR}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
