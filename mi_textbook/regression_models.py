#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
回帰モデルモジュール
=============================================================================

【学習目標】
    - 回帰分析の基本概念と各種手法を理解する
    - scikit-learnを用いた回帰モデルの実装方法を習得する
    - モデルの評価指標と過学習の概念を学ぶ

【前提知識】
    - 線形代数の基礎
    - 統計学の基礎（最小二乗法）
    - Python/NumPyの基本操作

【対象】
    材料工学部 3回生

【回帰分析とは】
    回帰分析は、入力変数（特徴量）から連続値の出力変数（目的変数）を
    予測する手法です。材料工学では、組成や構造から物性値を予測する
    場面で広く使われます。

【材料工学での応用例】
    - 組成から機械的強度を予測
    - 結晶構造からバンドギャップを予測
    - 製造条件から材料特性を予測

=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# 評価指標計算
# =============================================================================

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    回帰モデルの評価指標を計算する。

    【主な評価指標】
    - R²（決定係数）: 1に近いほど良い。モデルがデータの分散をどれだけ説明できるか
    - RMSE（二乗平均平方根誤差）: 0に近いほど良い。予測誤差の大きさ
    - MAE（平均絶対誤差）: 0に近いほど良い。外れ値に対してロバスト

    Args:
        y_true: 実測値
        y_pred: 予測値

    Returns:
        評価指標を含む辞書
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


# =============================================================================
# 線形回帰モデル
# =============================================================================

def train_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    線形回帰モデルを学習する。

    線形回帰は最も基本的な回帰手法で、以下の式でモデル化します：
    y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

    【特徴】
    - 解釈性が高い（係数の意味が明確）
    - 計算が高速
    - 非線形関係は捉えられない

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = LinearRegression()
    model.fit(x_train, y_train)

    results = {
        'model': model,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'train_predictions': model.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, model.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = model.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, model.predict(x_test)
        )

    return results


def train_polynomial_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    degree: int = 2,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    多項式回帰モデルを学習する。

    多項式回帰は、特徴量の多項式項を追加することで非線形関係を捉えます。
    例えば、degree=2の場合：y = w₀ + w₁x + w₂x²

    【注意点】
    - 次数が高すぎると過学習のリスク
    - 特徴量が多い場合は次元の爆発に注意

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        degree: 多項式の次数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    # パイプラインで多項式変換と線形回帰を組み合わせ
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])

    pipeline.fit(x_train, y_train)

    results = {
        'model': pipeline,
        'degree': degree,
        'train_predictions': pipeline.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, pipeline.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = pipeline.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, pipeline.predict(x_test)
        )

    return results


# =============================================================================
# 正則化回帰モデル
# =============================================================================

def train_ridge_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Ridge回帰（L2正則化）モデルを学習する。

    Ridge回帰は、損失関数にL2ペナルティ項を追加することで
    過学習を抑制します。係数の大きさを抑える効果があります。

    【損失関数】
    L = Σ(y - ŷ)² + α × Σw²

    【alphaの選び方】
    - 大きいほど正則化が強い（係数が小さくなる）
    - 交差検証で最適値を探索することが多い

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        alpha: 正則化パラメータ
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    results = {
        'model': model,
        'alpha': alpha,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'train_predictions': model.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, model.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = model.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, model.predict(x_test)
        )

    return results


def train_lasso_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Lasso回帰（L1正則化）モデルを学習する。

    Lasso回帰は、損失関数にL1ペナルティ項を追加します。
    特徴選択の効果があり、不要な特徴量の係数を0にします。

    【損失関数】
    L = Σ(y - ŷ)² + α × Σ|w|

    【Ridge vs Lasso】
    - Ridge: 全ての係数を小さくする
    - Lasso: 一部の係数を完全に0にする（スパース解）

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        alpha: 正則化パラメータ
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(x_train, y_train)

    results = {
        'model': model,
        'alpha': alpha,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'n_nonzero_coefficients': np.sum(model.coef_ != 0),
        'train_predictions': model.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, model.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = model.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, model.predict(x_test)
        )

    return results


# =============================================================================
# 非線形回帰モデル
# =============================================================================

def train_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = 'rbf',
    c_param: float = 1.0,
    epsilon: float = 0.1,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    サポートベクター回帰（SVR）モデルを学習する。

    SVRは、カーネル関数を用いて非線形関係を捉えます。
    マージン内の誤差を許容するε-insensitive損失を使用します。

    【カーネルの種類】
    - 'linear': 線形カーネル
    - 'rbf': RBFカーネル（ガウシアン）- 最も一般的
    - 'poly': 多項式カーネル

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        kernel: カーネル関数の種類
        c_param: 正則化パラメータ
        epsilon: ε-tubeの幅
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    # 標準化パイプライン（SVRは標準化が重要）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel=kernel, C=c_param, epsilon=epsilon))
    ])

    pipeline.fit(x_train, y_train)

    results = {
        'model': pipeline,
        'kernel': kernel,
        'c_param': c_param,
        'epsilon': epsilon,
        'train_predictions': pipeline.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, pipeline.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = pipeline.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, pipeline.predict(x_test)
        )

    return results


def train_decision_tree_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    決定木回帰モデルを学習する。

    決定木は、特徴量の閾値で分岐を繰り返し、葉ノードで予測値を出力します。
    解釈性が高く、非線形関係も捉えられます。

    【ハイパーパラメータ】
    - max_depth: 木の最大深さ（過学習防止）
    - min_samples_split: 分割に必要な最小サンプル数

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        max_depth: 木の最大深さ
        min_samples_split: 分割に必要な最小サンプル数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(x_train, y_train)

    results = {
        'model': model,
        'max_depth': max_depth,
        'feature_importances': model.feature_importances_,
        'train_predictions': model.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, model.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = model.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, model.predict(x_test)
        )

    return results


def train_random_forest_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    ランダムフォレスト回帰モデルを学習する。

    ランダムフォレストは、複数の決定木を組み合わせたアンサンブル手法です。
    各木はブートストラップサンプルと特徴量のサブセットで学習します。

    【特徴】
    - 過学習しにくい
    - 特徴量の重要度を算出可能
    - 並列計算が可能

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        n_estimators: 決定木の数
        max_depth: 各木の最大深さ
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    results = {
        'model': model,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_importances': model.feature_importances_,
        'train_predictions': model.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, model.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = model.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, model.predict(x_test)
        )

    return results


def train_neural_network_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (100,),
    max_iter: int = 1000,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    ニューラルネットワーク回帰モデルを学習する。

    多層パーセプトロン（MLP）は、複数の隠れ層を持つニューラルネットワークです。
    複雑な非線形関係を学習できますが、ハイパーパラメータの調整が必要です。

    【ハイパーパラメータ】
    - hidden_layer_sizes: 各隠れ層のユニット数
    - max_iter: 最大イテレーション数

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        hidden_layer_sizes: 隠れ層の構造
        max_iter: 最大イテレーション数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータの目的変数（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    # 標準化パイプライン（NNは標準化が重要）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True
        ))
    ])

    pipeline.fit(x_train, y_train)

    results = {
        'model': pipeline,
        'hidden_layer_sizes': hidden_layer_sizes,
        'train_predictions': pipeline.predict(x_train),
        'train_metrics': calculate_regression_metrics(
            y_train, pipeline.predict(x_train)
        )
    }

    if x_test is not None and y_test is not None:
        results['test_predictions'] = pipeline.predict(x_test)
        results['test_metrics'] = calculate_regression_metrics(
            y_test, pipeline.predict(x_test)
        )

    return results


# =============================================================================
# モデル比較
# =============================================================================

def compare_regression_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    複数の回帰モデルを比較する。

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        x_test: テストデータの特徴量
        y_test: テストデータの目的変数
        models: 比較するモデル名のリスト

    Returns:
        モデル比較結果のDataFrame
    """
    if models is None:
        models = [
            'linear', 'ridge', 'lasso', 'svr',
            'decision_tree', 'random_forest', 'neural_network'
        ]

    results_list = []

    model_functions = {
        'linear': lambda: train_linear_regression(
            x_train, y_train, x_test, y_test
        ),
        'ridge': lambda: train_ridge_regression(
            x_train, y_train, 1.0, x_test, y_test
        ),
        'lasso': lambda: train_lasso_regression(
            x_train, y_train, 0.1, x_test, y_test
        ),
        'svr': lambda: train_svr(
            x_train, y_train, 'rbf', 1.0, 0.1, x_test, y_test
        ),
        'decision_tree': lambda: train_decision_tree_regressor(
            x_train, y_train, 5, 2, x_test, y_test
        ),
        'random_forest': lambda: train_random_forest_regressor(
            x_train, y_train, 100, None, x_test, y_test
        ),
        'neural_network': lambda: train_neural_network_regressor(
            x_train, y_train, (50, 25), 500, x_test, y_test
        )
    }

    for model_name in models:
        if model_name in model_functions:
            result = model_functions[model_name]()
            results_list.append({
                'model': model_name,
                'train_r2': result['train_metrics']['r2'],
                'train_rmse': result['train_metrics']['rmse'],
                'test_r2': result['test_metrics']['r2'],
                'test_rmse': result['test_metrics']['rmse']
            })

    return pd.DataFrame(results_list)


# =============================================================================
# 可視化関数
# =============================================================================

def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Results",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    回帰結果を可視化する（実測値 vs 予測値）。

    Args:
        y_true: 実測値
        y_pred: 予測値
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 散布図
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='white', linewidth=0.5)

    # 理想線（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

    # 評価指標を表示
    metrics = calculate_regression_metrics(y_true, y_pred)
    text = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}"
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def demonstrate_overfitting(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degrees: List[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    多項式回帰で過学習を可視化する。

    次数を上げすぎると訓練データに過剰に適合し、
    新しいデータに対する予測性能が低下します（過学習）。

    Args:
        x_data: 特徴量データ（1次元）
        y_data: 目的変数データ
        degrees: 試す多項式の次数リスト
        save_path: 保存先パス

    Returns:
        Matplotlibの図オブジェクト
    """
    if degrees is None:
        degrees = [1, 3, 5, 15]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x_plot = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)

    for ax, degree in zip(axes, degrees):
        # モデル学習
        result = train_polynomial_regression(
            x_data.reshape(-1, 1), y_data, degree=degree
        )
        model = result['model']

        # 予測
        y_plot = model.predict(x_plot)

        # プロット
        ax.scatter(x_data, y_data, alpha=0.7, label='Data')
        ax.plot(x_plot, y_plot, 'r-', label=f'Degree {degree}')
        ax.set_title(f"Polynomial Degree = {degree}\n"
                     f"Train R² = {result['train_metrics']['r2']:.4f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
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
    print("回帰モデルモジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    np.random.seed(42)
    n_samples = 200

    # 非線形関係を持つデータを生成
    x_data = np.random.uniform(0, 10, (n_samples, 3))
    y_data = (
        2.0 * x_data[:, 0]
        + 0.5 * x_data[:, 1] ** 2
        - 1.0 * x_data[:, 2]
        + np.random.normal(0, 1, n_samples)
    )

    # 訓練・テスト分割
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    print(f"訓練データ: {x_train.shape[0]}サンプル")
    print(f"テストデータ: {x_test.shape[0]}サンプル")

    # 各モデルの学習と評価
    print("\n【2. 各モデルの学習と評価】")
    print("-" * 50)

    # 線形回帰
    print("\n[線形回帰]")
    lr_result = train_linear_regression(x_train, y_train, x_test, y_test)
    print(f"  訓練R²: {lr_result['train_metrics']['r2']:.4f}")
    print(f"  テストR²: {lr_result['test_metrics']['r2']:.4f}")

    # Ridge回帰
    print("\n[Ridge回帰]")
    ridge_result = train_ridge_regression(x_train, y_train, 1.0, x_test, y_test)
    print(f"  訓練R²: {ridge_result['train_metrics']['r2']:.4f}")
    print(f"  テストR²: {ridge_result['test_metrics']['r2']:.4f}")

    # ランダムフォレスト
    print("\n[ランダムフォレスト]")
    rf_result = train_random_forest_regressor(
        x_train, y_train, 100, None, x_test, y_test
    )
    print(f"  訓練R²: {rf_result['train_metrics']['r2']:.4f}")
    print(f"  テストR²: {rf_result['test_metrics']['r2']:.4f}")

    # モデル比較
    print("\n【3. モデル比較】")
    print("-" * 50)

    comparison_df = compare_regression_models(x_train, y_train, x_test, y_test)
    print(comparison_df.to_string(index=False))

    # 可視化
    print("\n【4. 可視化】")
    print("-" * 50)

    fig = plot_regression_results(
        y_test, rf_result['test_predictions'],
        title="Random Forest Regression Results"
    )
    plt.close(fig)
    print("回帰結果プロット: 作成完了")

    print("\n処理完了!")
