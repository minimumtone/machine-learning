#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ベイズ最適化・ガウス過程回帰モジュール
=============================================================================

【学習目標】
    - ガウス過程回帰（GPR）の基本概念を理解する
    - ベイズ最適化の原理と応用を学ぶ
    - 材料探索への応用方法を習得する

【前提知識】
    - 確率・統計の基礎（正規分布、ベイズの定理）
    - 回帰分析の基本概念
    - Python/NumPyの基本操作

【対象】
    材料工学部 3回生

【ガウス過程回帰とは】
    ガウス過程回帰（Gaussian Process Regression, GPR）は、
    関数の事前分布を仮定し、観測データから事後分布を推定する手法です。
    予測値だけでなく、予測の不確実性も同時に得られます。

【ベイズ最適化とは】
    ベイズ最適化は、評価コストの高い関数の最適化に適した手法です。
    GPRで関数を近似し、獲得関数を最大化する点を次の評価点として選びます。

【材料工学での応用例】
    - 実験条件の最適化
    - 材料組成の最適化
    - プロセスパラメータの最適化

=============================================================================
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
)


# =============================================================================
# ガウス過程回帰
# =============================================================================

def train_gpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = 'rbf',
    alpha: float = 1e-10,
    normalize_y: bool = True
) -> Dict[str, Any]:
    """
    ガウス過程回帰モデルを学習する。

    GPRは以下の特徴を持ちます：
    - 予測の不確実性（分散）を同時に推定
    - カーネル関数で関数の滑らかさを制御
    - 少ないデータでも有効

    【カーネル関数】
    - RBF（Radial Basis Function）: 滑らかな関数を仮定
    - Matern: RBFより柔軟、微分可能性を制御可能

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        kernel: カーネル関数の種類 ('rbf', 'matern')
        alpha: ノイズレベル
        normalize_y: 目的変数を正規化するかどうか

    Returns:
        学習済みモデルと関連情報を含む辞書

    Example:
        >>> results = train_gpr(x_train, y_train, kernel='rbf')
        >>> y_pred, y_std = results['model'].predict(x_test, return_std=True)
    """
    # カーネル関数の設定
    if kernel == 'rbf':
        kernel_func = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()
    elif kernel == 'matern':
        kernel_func = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel()
    else:
        kernel_func = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()

    # GPRモデルの学習
    model = GaussianProcessRegressor(
        kernel=kernel_func,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=10,
        random_state=42
    )
    model.fit(x_train, y_train)

    # 訓練データでの予測
    y_pred, y_std = model.predict(x_train, return_std=True)

    return {
        'model': model,
        'kernel': str(model.kernel_),
        'log_marginal_likelihood': model.log_marginal_likelihood_value_,
        'train_predictions': y_pred,
        'train_std': y_std
    }


def predict_with_uncertainty(
    model: GaussianProcessRegressor,
    x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GPRモデルで予測と信頼区間を計算する。

    Args:
        model: 学習済みGPRモデル
        x_test: テストデータ

    Returns:
        予測平均、予測標準偏差、95%信頼区間の下限・上限のタプル
    """
    y_pred, y_std = model.predict(x_test, return_std=True)

    # 95%信頼区間
    ci_lower = y_pred - 1.96 * y_std
    ci_upper = y_pred + 1.96 * y_std

    return y_pred, y_std, ci_lower, ci_upper


# =============================================================================
# 獲得関数
# =============================================================================

def expected_improvement(
    x: np.ndarray,
    model: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01
) -> float:
    """
    Expected Improvement（EI）獲得関数を計算する。

    EIは、現在の最良値からの改善量の期待値を計算します。
    探索（不確実性の高い領域）と活用（予測値の高い領域）の
    バランスを取ります。

    【数式】
    EI(x) = (μ(x) - y_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    Z = (μ(x) - y_best - ξ) / σ(x)

    Args:
        x: 評価点
        model: 学習済みGPRモデル
        y_best: 現在の最良値
        xi: 探索パラメータ（大きいほど探索重視）

    Returns:
        EI値
    """
    x = np.atleast_2d(x)
    mu, sigma = model.predict(x, return_std=True)

    if sigma == 0:
        return 0.0

    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    return ei[0]


def probability_of_improvement(
    x: np.ndarray,
    model: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01
) -> float:
    """
    Probability of Improvement（PI）獲得関数を計算する。

    PIは、現在の最良値を超える確率を計算します。
    EIより単純ですが、改善量を考慮しません。

    Args:
        x: 評価点
        model: 学習済みGPRモデル
        y_best: 現在の最良値
        xi: 探索パラメータ

    Returns:
        PI値
    """
    x = np.atleast_2d(x)
    mu, sigma = model.predict(x, return_std=True)

    if sigma == 0:
        return 0.0

    z = (mu - y_best - xi) / sigma
    return norm.cdf(z)[0]


def upper_confidence_bound(
    x: np.ndarray,
    model: GaussianProcessRegressor,
    kappa: float = 2.0
) -> float:
    """
    Upper Confidence Bound（UCB）獲得関数を計算する。

    UCBは、予測平均と不確実性の重み付き和を計算します。
    kappaで探索と活用のバランスを制御します。

    【数式】
    UCB(x) = μ(x) + κ * σ(x)

    Args:
        x: 評価点
        model: 学習済みGPRモデル
        kappa: 探索パラメータ（大きいほど探索重視）

    Returns:
        UCB値
    """
    x = np.atleast_2d(x)
    mu, sigma = model.predict(x, return_std=True)

    return (mu + kappa * sigma)[0]


# =============================================================================
# ベイズ最適化
# =============================================================================

def bayesian_optimization(
    objective_func: Callable,
    bounds: np.ndarray,
    n_initial: int = 5,
    n_iterations: int = 20,
    acquisition: str = 'ei',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    ベイズ最適化を実行する。

    ベイズ最適化は以下の手順で進みます：
    1. 初期点をランダムにサンプリングして評価
    2. GPRで関数を近似
    3. 獲得関数を最大化する点を次の評価点として選択
    4. 2-3を繰り返し

    【獲得関数の選択】
    - 'ei': Expected Improvement（最も一般的）
    - 'pi': Probability of Improvement
    - 'ucb': Upper Confidence Bound

    Args:
        objective_func: 最適化する目的関数
        bounds: 探索範囲 [[x1_min, x1_max], [x2_min, x2_max], ...]
        n_initial: 初期サンプル数
        n_iterations: 最適化イテレーション数
        acquisition: 獲得関数の種類
        random_state: 乱数シード

    Returns:
        最適化結果を含む辞書

    Example:
        >>> def objective(x):
        ...     return -((x[0] - 2)**2 + (x[1] - 1)**2)
        >>> bounds = np.array([[0, 5], [0, 5]])
        >>> results = bayesian_optimization(objective, bounds, n_iterations=30)
        >>> print(f"最適解: {results['best_x']}")
    """
    np.random.seed(random_state)

    n_dims = len(bounds)

    # 初期サンプリング（ラテン超方格サンプリングの簡易版）
    x_samples = np.random.uniform(
        bounds[:, 0], bounds[:, 1],
        size=(n_initial, n_dims)
    )
    y_samples = np.array([objective_func(x) for x in x_samples])

    # 最適化履歴
    history = {
        'x': list(x_samples),
        'y': list(y_samples),
        'best_y': [y_samples.max()],
        'best_x': [x_samples[y_samples.argmax()]]
    }

    # ベイズ最適化ループ
    for i in range(n_iterations):
        # GPRモデルの学習
        gpr_result = train_gpr(
            np.array(history['x']),
            np.array(history['y']),
            kernel='matern'
        )
        model = gpr_result['model']

        # 現在の最良値
        y_best = max(history['y'])

        # 獲得関数の最大化
        def neg_acquisition(x):
            if acquisition == 'ei':
                return -expected_improvement(x, model, y_best)
            elif acquisition == 'pi':
                return -probability_of_improvement(x, model, y_best)
            elif acquisition == 'ucb':
                return -upper_confidence_bound(x, model)
            else:
                return -expected_improvement(x, model, y_best)

        # 複数の初期点から最適化
        best_acq = -np.inf
        best_x = None

        for _ in range(10):
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            result = minimize(
                neg_acquisition,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x

        # 新しい点を評価
        y_new = objective_func(best_x)

        # 履歴を更新
        history['x'].append(best_x)
        history['y'].append(y_new)

        if y_new > max(history['best_y']):
            history['best_y'].append(y_new)
            history['best_x'].append(best_x)
        else:
            history['best_y'].append(history['best_y'][-1])
            history['best_x'].append(history['best_x'][-1])

    # 最終結果
    best_idx = np.argmax(history['y'])

    return {
        'best_x': history['x'][best_idx],
        'best_y': history['y'][best_idx],
        'history': history,
        'n_evaluations': n_initial + n_iterations
    }


# =============================================================================
# 可視化関数
# =============================================================================

def plot_gpr_1d(
    model: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_range: Tuple[float, float] = (0, 10),
    n_points: int = 100,
    true_func: Optional[Callable] = None,
    title: str = "Gaussian Process Regression",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    1次元GPRの結果を可視化する。

    Args:
        model: 学習済みGPRモデル
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        x_range: 表示範囲
        n_points: プロット点数
        true_func: 真の関数（オプション）
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 予測用のグリッド
    x_plot = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
    y_pred, y_std, ci_lower, ci_upper = predict_with_uncertainty(model, x_plot)

    # 予測平均と信頼区間
    ax.plot(x_plot, y_pred, 'b-', label='GPR Mean')
    ax.fill_between(
        x_plot.flatten(),
        ci_lower, ci_upper,
        alpha=0.3, color='blue',
        label='95% Confidence Interval'
    )

    # 訓練データ
    ax.scatter(x_train, y_train, c='red', s=50, zorder=5, label='Training Data')

    # 真の関数
    if true_func is not None:
        y_true = np.array([true_func(x) for x in x_plot])
        ax.plot(x_plot, y_true, 'g--', label='True Function')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_optimization_history(
    history: Dict[str, List],
    title: str = "Bayesian Optimization History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    ベイズ最適化の履歴を可視化する。

    Args:
        history: 最適化履歴
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    iterations = range(1, len(history['y']) + 1)

    # 各イテレーションの目的関数値
    axes[0].plot(iterations, history['y'], 'bo-', alpha=0.7)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('Objective Value per Iteration')
    axes[0].grid(True, alpha=0.3)

    # 最良値の推移
    axes[1].plot(iterations, history['best_y'], 'go-')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Best Objective Value')
    axes[1].set_title('Best Value Found')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_acquisition_function(
    model: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_range: Tuple[float, float] = (0, 10),
    acquisition: str = 'ei',
    title: str = "Acquisition Function",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    獲得関数を可視化する。

    Args:
        model: 学習済みGPRモデル
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        x_range: 表示範囲
        acquisition: 獲得関数の種類
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    x_plot = np.linspace(x_range[0], x_range[1], 100).reshape(-1, 1)
    y_pred, y_std, ci_lower, ci_upper = predict_with_uncertainty(model, x_plot)

    y_best = y_train.max()

    # GPR予測
    axes[0].plot(x_plot, y_pred, 'b-', label='GPR Mean')
    axes[0].fill_between(
        x_plot.flatten(),
        ci_lower, ci_upper,
        alpha=0.3, color='blue'
    )
    axes[0].scatter(x_train, y_train, c='red', s=50, zorder=5, label='Data')
    axes[0].axhline(y=y_best, color='green', linestyle='--', label=f'Best: {y_best:.2f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('GPR Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 獲得関数
    if acquisition == 'ei':
        acq_values = [expected_improvement(x, model, y_best) for x in x_plot]
        acq_name = 'Expected Improvement'
    elif acquisition == 'pi':
        acq_values = [probability_of_improvement(x, model, y_best) for x in x_plot]
        acq_name = 'Probability of Improvement'
    else:
        acq_values = [upper_confidence_bound(x, model) for x in x_plot]
        acq_name = 'Upper Confidence Bound'

    axes[1].plot(x_plot, acq_values, 'purple', linewidth=2)
    axes[1].fill_between(x_plot.flatten(), 0, acq_values, alpha=0.3, color='purple')
    next_x = x_plot[np.argmax(acq_values)]
    axes[1].axvline(x=next_x, color='red', linestyle='--', label=f'Next: x={next_x[0]:.2f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel(acq_name)
    axes[1].set_title(f'Acquisition Function ({acq_name})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ベイズ最適化・ガウス過程回帰モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. ガウス過程回帰のデモ】")
    print("-" * 50)

    # 真の関数
    def true_function(x):
        return np.sin(x) + 0.5 * np.sin(3 * x)

    np.random.seed(42)
    x_train = np.random.uniform(0, 10, 10).reshape(-1, 1)
    y_train = np.array([true_function(x) for x in x_train]).flatten()
    y_train += np.random.normal(0, 0.1, len(y_train))  # ノイズ追加

    # GPRの学習
    gpr_result = train_gpr(x_train, y_train, kernel='rbf')
    print(f"学習済みカーネル: {gpr_result['kernel']}")
    print(f"対数周辺尤度: {gpr_result['log_marginal_likelihood']:.4f}")

    # ベイズ最適化のデモ
    print("\n【2. ベイズ最適化のデモ】")
    print("-" * 50)

    # 最適化する目的関数（最大化）
    def objective(x):
        return -((x[0] - 3)**2 + (x[1] - 2)**2) + 10

    bounds = np.array([[0, 6], [0, 6]])

    bo_result = bayesian_optimization(
        objective,
        bounds,
        n_initial=5,
        n_iterations=15,
        acquisition='ei'
    )

    print(f"最適解: x = {bo_result['best_x']}")
    print(f"最適値: y = {bo_result['best_y']:.4f}")
    print(f"評価回数: {bo_result['n_evaluations']}")

    # 可視化
    print("\n【3. 可視化】")
    print("-" * 50)

    fig1 = plot_gpr_1d(
        gpr_result['model'],
        x_train, y_train,
        x_range=(0, 10),
        true_func=true_function,
        title="GPR with Uncertainty"
    )
    plt.close(fig1)
    print("GPR可視化: 作成完了")

    fig2 = plot_optimization_history(
        bo_result['history'],
        title="Bayesian Optimization Progress"
    )
    plt.close(fig2)
    print("最適化履歴: 作成完了")

    print("\n処理完了!")
