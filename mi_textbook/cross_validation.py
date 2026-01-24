#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
交差検証モジュール
=============================================================================

【学習目標】
    - 交差検証の目的と重要性を理解する
    - 各種交差検証手法の特徴と使い分けを学ぶ
    - ハイパーパラメータチューニングの方法を習得する

【前提知識】
    - 機械学習の基本概念（訓練、テスト、汎化）
    - 過学習の概念
    - Python/scikit-learnの基本操作

【対象】
    材料工学部 3回生

【交差検証とは】
    交差検証は、モデルの汎化性能を評価するための手法です。
    データを複数の部分に分割し、それぞれを訓練とテストに使用することで、
    より信頼性の高い性能評価が可能になります。

【なぜ交差検証が必要か】
    - 単一の訓練/テスト分割では、分割の仕方に結果が依存
    - データが少ない場合、テストデータが不足
    - 過学習の検出に有効

=============================================================================
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    learning_curve,
)


# =============================================================================
# 基本的な交差検証
# =============================================================================

def perform_kfold_cv(
    model: Any,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_splits: int = 5,
    scoring: str = 'r2',
    shuffle: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    K分割交差検証を実行する。

    K分割交差検証では、データをK個の部分（フォールド）に分割し、
    K-1個で訓練、1個でテストを行います。これをK回繰り返し、
    全てのデータがテストに使われます。

    【Kの選び方】
    - 一般的にはK=5または10
    - データが少ない場合はKを大きく（最大でLOOCV）
    - 計算コストとのトレードオフ

    Args:
        model: 学習モデル
        x_data: 特徴量データ
        y_data: 目的変数データ
        n_splits: 分割数K
        scoring: 評価指標（'r2', 'neg_mean_squared_error', 'accuracy'など）
        shuffle: データをシャッフルするかどうか
        random_state: 乱数シード

    Returns:
        交差検証結果を含む辞書

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> results = perform_kfold_cv(model, x_data, y_data, n_splits=5)
        >>> print(f"平均スコア: {results['mean_score']:.4f}")
    """
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scores = cross_val_score(model, x_data, y_data, cv=kfold, scoring=scoring)

    return {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'n_splits': n_splits,
        'scoring': scoring
    }


def perform_stratified_kfold_cv(
    model: Any,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_splits: int = 5,
    scoring: str = 'accuracy',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    層化K分割交差検証を実行する。

    層化K分割交差検証は、各フォールドでクラスの比率が
    元のデータと同じになるように分割します。
    クラス不均衡なデータに特に有効です。

    Args:
        model: 学習モデル
        x_data: 特徴量データ
        y_data: 目的変数データ（カテゴリカル）
        n_splits: 分割数K
        scoring: 評価指標
        random_state: 乱数シード

    Returns:
        交差検証結果を含む辞書
    """
    skfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(model, x_data, y_data, cv=skfold, scoring=scoring)

    return {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'n_splits': n_splits,
        'scoring': scoring
    }


def perform_loocv(
    model: Any,
    x_data: np.ndarray,
    y_data: np.ndarray,
    scoring: str = 'r2'
) -> Dict[str, Any]:
    """
    Leave-One-Out交差検証（LOOCV）を実行する。

    LOOCVは、1つのサンプルをテストに、残り全てを訓練に使用します。
    これをサンプル数だけ繰り返します。

    【特徴】
    - バイアスが最小（訓練データが最大）
    - 分散が大きい可能性
    - 計算コストが高い（サンプル数回の学習）

    Args:
        model: 学習モデル
        x_data: 特徴量データ
        y_data: 目的変数データ
        scoring: 評価指標

    Returns:
        交差検証結果を含む辞書
    """
    loo = LeaveOneOut()

    scores = cross_val_score(model, x_data, y_data, cv=loo, scoring=scoring)

    return {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'n_samples': len(y_data),
        'scoring': scoring
    }


def perform_detailed_cv(
    model: Any,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_splits: int = 5,
    scoring: Union[str, List[str]] = None,
    return_train_score: bool = True
) -> Dict[str, Any]:
    """
    詳細な交差検証結果を取得する。

    cross_validateを使用して、複数の評価指標と
    訓練スコアも含めた詳細な結果を取得します。

    Args:
        model: 学習モデル
        x_data: 特徴量データ
        y_data: 目的変数データ
        n_splits: 分割数
        scoring: 評価指標（リストで複数指定可能）
        return_train_score: 訓練スコアも返すかどうか

    Returns:
        詳細な交差検証結果を含む辞書
    """
    if scoring is None:
        scoring = ['r2', 'neg_mean_squared_error']

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, x_data, y_data,
        cv=kfold,
        scoring=scoring,
        return_train_score=return_train_score
    )

    return cv_results


# =============================================================================
# ハイパーパラメータチューニング
# =============================================================================

def perform_grid_search(
    model: Any,
    param_grid: Dict[str, List],
    x_data: np.ndarray,
    y_data: np.ndarray,
    cv: int = 5,
    scoring: str = 'r2',
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    グリッドサーチでハイパーパラメータを最適化する。

    グリッドサーチは、指定したパラメータの全組み合わせを
    試して最適な組み合わせを見つけます。

    【注意点】
    - パラメータ数が多いと計算コストが爆発的に増加
    - 粗いグリッドで探索後、細かいグリッドで再探索が効果的

    Args:
        model: 学習モデル
        param_grid: パラメータグリッド
        x_data: 特徴量データ
        y_data: 目的変数データ
        cv: 交差検証の分割数
        scoring: 評価指標
        n_jobs: 並列実行数（-1で全コア使用）

    Returns:
        グリッドサーチ結果を含む辞書

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> param_grid = {
        ...     'n_estimators': [50, 100, 200],
        ...     'max_depth': [3, 5, 10, None]
        ... }
        >>> results = perform_grid_search(
        ...     RandomForestRegressor(), param_grid, x_data, y_data
        ... )
        >>> print(f"最適パラメータ: {results['best_params']}")
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True
    )

    grid_search.fit(x_data, y_data)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }


def perform_random_search(
    model: Any,
    param_distributions: Dict[str, Any],
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_iter: int = 100,
    cv: int = 5,
    scoring: str = 'r2',
    n_jobs: int = -1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    ランダムサーチでハイパーパラメータを最適化する。

    ランダムサーチは、パラメータ空間からランダムにサンプリングして
    最適な組み合わせを探索します。グリッドサーチより効率的な場合が多いです。

    【グリッドサーチとの比較】
    - 計算コストを制御しやすい（n_iterで指定）
    - 連続値パラメータに対して効果的
    - 重要でないパラメータに計算資源を浪費しない

    Args:
        model: 学習モデル
        param_distributions: パラメータ分布
        x_data: 特徴量データ
        y_data: 目的変数データ
        n_iter: サンプリング回数
        cv: 交差検証の分割数
        scoring: 評価指標
        n_jobs: 並列実行数
        random_state: 乱数シード

    Returns:
        ランダムサーチ結果を含む辞書
    """
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=True
    )

    random_search.fit(x_data, y_data)

    return {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'best_model': random_search.best_estimator_,
        'cv_results': pd.DataFrame(random_search.cv_results_)
    }


# =============================================================================
# 学習曲線
# =============================================================================

def compute_learning_curve(
    model: Any,
    x_data: np.ndarray,
    y_data: np.ndarray,
    train_sizes: np.ndarray = None,
    cv: int = 5,
    scoring: str = 'r2'
) -> Dict[str, Any]:
    """
    学習曲線を計算する。

    学習曲線は、訓練データ数に対するモデル性能の変化を示します。
    過学習や未学習の診断に有用です。

    【学習曲線の解釈】
    - 訓練スコアとテストスコアが両方低い: 未学習（モデルが単純すぎる）
    - 訓練スコアが高くテストスコアが低い: 過学習
    - 両方が高く収束: 良好なモデル

    Args:
        model: 学習モデル
        x_data: 特徴量データ
        y_data: 目的変数データ
        train_sizes: 訓練データサイズの配列
        cv: 交差検証の分割数
        scoring: 評価指標

    Returns:
        学習曲線データを含む辞書
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, x_data, y_data,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': train_scores.mean(axis=1),
        'train_scores_std': train_scores.std(axis=1),
        'test_scores_mean': test_scores.mean(axis=1),
        'test_scores_std': test_scores.std(axis=1)
    }


# =============================================================================
# 可視化関数
# =============================================================================

def plot_cv_scores(
    cv_results: Dict[str, Any],
    title: str = "Cross-Validation Scores",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    交差検証スコアを可視化する。

    Args:
        cv_results: 交差検証結果
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    scores = cv_results['scores']
    folds = range(1, len(scores) + 1)

    ax.bar(folds, scores, alpha=0.7)
    ax.axhline(
        y=cv_results['mean_score'],
        color='r', linestyle='--',
        label=f"Mean: {cv_results['mean_score']:.4f}"
    )
    ax.fill_between(
        [0.5, len(scores) + 0.5],
        cv_results['mean_score'] - cv_results['std_score'],
        cv_results['mean_score'] + cv_results['std_score'],
        alpha=0.2, color='r',
        label=f"Std: {cv_results['std_score']:.4f}"
    )

    ax.set_xlabel('Fold')
    ax.set_ylabel(cv_results['scoring'])
    ax.set_title(title)
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_learning_curve(
    lc_results: Dict[str, Any],
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    学習曲線を可視化する。

    Args:
        lc_results: 学習曲線データ
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_sizes = lc_results['train_sizes']
    train_mean = lc_results['train_scores_mean']
    train_std = lc_results['train_scores_std']
    test_mean = lc_results['test_scores_mean']
    test_std = lc_results['test_scores_std']

    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2, color='blue'
    )

    ax.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2, color='green'
    )

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_grid_search_results(
    gs_results: Dict[str, Any],
    param_name: str,
    title: str = "Grid Search Results",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    グリッドサーチ結果を可視化する。

    Args:
        gs_results: グリッドサーチ結果
        param_name: 可視化するパラメータ名
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    cv_results = gs_results['cv_results']
    param_col = f'param_{param_name}'

    if param_col in cv_results.columns:
        param_values = cv_results[param_col].astype(str)
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']

        ax.errorbar(
            range(len(param_values)), mean_scores,
            yerr=std_scores, fmt='o-', capsize=5
        )
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels(param_values, rotation=45, ha='right')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("交差検証モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge

    x_data, y_data = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )

    print(f"データ形状: {x_data.shape}")

    # K分割交差検証
    print("\n【2. K分割交差検証】")
    print("-" * 50)

    model = Ridge(alpha=1.0)
    cv_results = perform_kfold_cv(model, x_data, y_data, n_splits=5)

    print(f"各フォールドのスコア: {cv_results['scores']}")
    print(f"平均スコア: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

    # グリッドサーチ
    print("\n【3. グリッドサーチ】")
    print("-" * 50)

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    gs_results = perform_grid_search(Ridge(), param_grid, x_data, y_data)

    print(f"最適パラメータ: {gs_results['best_params']}")
    print(f"最高スコア: {gs_results['best_score']:.4f}")

    # 学習曲線
    print("\n【4. 学習曲線】")
    print("-" * 50)

    lc_results = compute_learning_curve(Ridge(alpha=1.0), x_data, y_data)
    print(f"訓練サイズ: {lc_results['train_sizes']}")
    print(f"テストスコア（最終）: {lc_results['test_scores_mean'][-1]:.4f}")

    # 可視化
    print("\n【5. 可視化】")
    print("-" * 50)

    fig1 = plot_cv_scores(cv_results, title="5-Fold Cross-Validation")
    plt.close(fig1)
    print("交差検証スコアプロット: 作成完了")

    fig2 = plot_learning_curve(lc_results)
    plt.close(fig2)
    print("学習曲線プロット: 作成完了")

    print("\n処理完了!")
