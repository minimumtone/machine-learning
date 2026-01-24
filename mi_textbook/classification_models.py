#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
分類モデルモジュール
=============================================================================

【学習目標】
    - 分類問題の基本概念と各種手法を理解する
    - scikit-learnを用いた分類モデルの実装方法を習得する
    - 分類モデルの評価指標（精度、適合率、再現率、F1スコア）を学ぶ

【前提知識】
    - 確率・統計の基礎
    - 回帰分析の基本概念
    - Python/NumPyの基本操作

【対象】
    材料工学部 3回生

【分類問題とは】
    分類問題は、入力データを離散的なカテゴリ（クラス）に分類する問題です。
    回帰問題が連続値を予測するのに対し、分類問題は離散的なラベルを予測します。

【材料工学での応用例】
    - 材料の結晶構造の分類
    - 欠陥の有無の判定
    - 材料の安定性（安定/不安定）の予測

=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# 評価指標計算
# =============================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    分類モデルの評価指標を計算する。

    【主な評価指標】
    - Accuracy（精度）: 全体の正解率
    - Precision（適合率）: 陽性予測のうち実際に陽性の割合
    - Recall（再現率）: 実際の陽性のうち正しく予測できた割合
    - F1 Score: PrecisionとRecallの調和平均
    - AUC: ROC曲線の下の面積（確率予測が必要）

    Args:
        y_true: 実際のラベル
        y_pred: 予測ラベル
        y_prob: 予測確率（AUC計算用、オプション）

    Returns:
        評価指標を含む辞書
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # 二値分類の場合
                if y_prob.ndim == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
            else:
                # 多クラス分類の場合
                metrics['auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                )
        except ValueError:
            metrics['auc'] = None

    return metrics


# =============================================================================
# 分類モデル
# =============================================================================

def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_param: float = 1.0,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    ロジスティック回帰モデルを学習する。

    ロジスティック回帰は、線形モデルにシグモイド関数を適用して
    確率を出力する分類手法です。名前に「回帰」とありますが、分類に使います。

    【特徴】
    - 解釈性が高い（係数の意味が明確）
    - 確率を出力できる
    - 線形分離可能なデータに適する

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        c_param: 正則化パラメータの逆数（大きいほど正則化が弱い）
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータのラベル（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=c_param, max_iter=1000))
    ])

    pipeline.fit(x_train, y_train)

    train_pred = pipeline.predict(x_train)
    train_prob = pipeline.predict_proba(x_train)

    results = {
        'model': pipeline,
        'train_predictions': train_pred,
        'train_probabilities': train_prob,
        'train_metrics': calculate_classification_metrics(
            y_train, train_pred, train_prob
        )
    }

    if x_test is not None and y_test is not None:
        test_pred = pipeline.predict(x_test)
        test_prob = pipeline.predict_proba(x_test)
        results['test_predictions'] = test_pred
        results['test_probabilities'] = test_prob
        results['test_metrics'] = calculate_classification_metrics(
            y_test, test_pred, test_prob
        )

    return results


def train_svm_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = 'rbf',
    c_param: float = 1.0,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    サポートベクターマシン（SVM）分類モデルを学習する。

    SVMは、クラス間のマージンを最大化する決定境界を見つけます。
    カーネル関数を使うことで非線形な決定境界も学習できます。

    【カーネルの種類】
    - 'linear': 線形カーネル（線形分離可能なデータ向け）
    - 'rbf': RBFカーネル（最も一般的）
    - 'poly': 多項式カーネル

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        kernel: カーネル関数の種類
        c_param: 正則化パラメータ
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータのラベル（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel=kernel, C=c_param, probability=True))
    ])

    pipeline.fit(x_train, y_train)

    train_pred = pipeline.predict(x_train)
    train_prob = pipeline.predict_proba(x_train)

    results = {
        'model': pipeline,
        'kernel': kernel,
        'train_predictions': train_pred,
        'train_probabilities': train_prob,
        'train_metrics': calculate_classification_metrics(
            y_train, train_pred, train_prob
        )
    }

    if x_test is not None and y_test is not None:
        test_pred = pipeline.predict(x_test)
        test_prob = pipeline.predict_proba(x_test)
        results['test_predictions'] = test_pred
        results['test_probabilities'] = test_prob
        results['test_metrics'] = calculate_classification_metrics(
            y_test, test_pred, test_prob
        )

    return results


def train_knn_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    k近傍法（k-NN）分類モデルを学習する。

    k-NNは、新しいサンプルに最も近いk個の訓練サンプルの
    多数決でクラスを決定します。シンプルですが効果的な手法です。

    【kの選び方】
    - 小さいk: 決定境界が複雑（過学習のリスク）
    - 大きいk: 決定境界が滑らか（未学習のリスク）
    - 一般的には奇数を選ぶ（同票を避けるため）

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        n_neighbors: 近傍数k
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータのラベル（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])

    pipeline.fit(x_train, y_train)

    train_pred = pipeline.predict(x_train)
    train_prob = pipeline.predict_proba(x_train)

    results = {
        'model': pipeline,
        'n_neighbors': n_neighbors,
        'train_predictions': train_pred,
        'train_probabilities': train_prob,
        'train_metrics': calculate_classification_metrics(
            y_train, train_pred, train_prob
        )
    }

    if x_test is not None and y_test is not None:
        test_pred = pipeline.predict(x_test)
        test_prob = pipeline.predict_proba(x_test)
        results['test_predictions'] = test_pred
        results['test_probabilities'] = test_prob
        results['test_metrics'] = calculate_classification_metrics(
            y_test, test_pred, test_prob
        )

    return results


def train_decision_tree_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    決定木分類モデルを学習する。

    決定木は、特徴量の閾値で分岐を繰り返し、葉ノードでクラスを出力します。
    解釈性が高く、可視化も容易です。

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        max_depth: 木の最大深さ
        min_samples_split: 分割に必要な最小サンプル数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータのラベル（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    train_prob = model.predict_proba(x_train)

    results = {
        'model': model,
        'max_depth': max_depth,
        'feature_importances': model.feature_importances_,
        'train_predictions': train_pred,
        'train_probabilities': train_prob,
        'train_metrics': calculate_classification_metrics(
            y_train, train_pred, train_prob
        )
    }

    if x_test is not None and y_test is not None:
        test_pred = model.predict(x_test)
        test_prob = model.predict_proba(x_test)
        results['test_predictions'] = test_pred
        results['test_probabilities'] = test_prob
        results['test_metrics'] = calculate_classification_metrics(
            y_test, test_pred, test_prob
        )

    return results


def train_random_forest_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    ランダムフォレスト分類モデルを学習する。

    ランダムフォレストは、複数の決定木を組み合わせたアンサンブル手法です。
    各木の予測の多数決で最終的なクラスを決定します。

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        n_estimators: 決定木の数
        max_depth: 各木の最大深さ
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータのラベル（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    train_prob = model.predict_proba(x_train)

    results = {
        'model': model,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_importances': model.feature_importances_,
        'train_predictions': train_pred,
        'train_probabilities': train_prob,
        'train_metrics': calculate_classification_metrics(
            y_train, train_pred, train_prob
        )
    }

    if x_test is not None and y_test is not None:
        test_pred = model.predict(x_test)
        test_prob = model.predict_proba(x_test)
        results['test_predictions'] = test_pred
        results['test_probabilities'] = test_prob
        results['test_metrics'] = calculate_classification_metrics(
            y_test, test_pred, test_prob
        )

    return results


def train_neural_network_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (100,),
    max_iter: int = 1000,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    ニューラルネットワーク分類モデルを学習する。

    多層パーセプトロン（MLP）は、複数の隠れ層を持つニューラルネットワークです。
    複雑な非線形決定境界を学習できます。

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        hidden_layer_sizes: 隠れ層の構造
        max_iter: 最大イテレーション数
        x_test: テストデータの特徴量（オプション）
        y_test: テストデータのラベル（オプション）

    Returns:
        モデルと評価結果を含む辞書
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True
        ))
    ])

    pipeline.fit(x_train, y_train)

    train_pred = pipeline.predict(x_train)
    train_prob = pipeline.predict_proba(x_train)

    results = {
        'model': pipeline,
        'hidden_layer_sizes': hidden_layer_sizes,
        'train_predictions': train_pred,
        'train_probabilities': train_prob,
        'train_metrics': calculate_classification_metrics(
            y_train, train_pred, train_prob
        )
    }

    if x_test is not None and y_test is not None:
        test_pred = pipeline.predict(x_test)
        test_prob = pipeline.predict_proba(x_test)
        results['test_predictions'] = test_pred
        results['test_probabilities'] = test_prob
        results['test_metrics'] = calculate_classification_metrics(
            y_test, test_pred, test_prob
        )

    return results


# =============================================================================
# モデル比較
# =============================================================================

def compare_classification_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    複数の分類モデルを比較する。

    Args:
        x_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        x_test: テストデータの特徴量
        y_test: テストデータのラベル
        models: 比較するモデル名のリスト

    Returns:
        モデル比較結果のDataFrame
    """
    if models is None:
        models = [
            'logistic', 'svm', 'knn',
            'decision_tree', 'random_forest', 'neural_network'
        ]

    results_list = []

    model_functions = {
        'logistic': lambda: train_logistic_regression(
            x_train, y_train, 1.0, x_test, y_test
        ),
        'svm': lambda: train_svm_classifier(
            x_train, y_train, 'rbf', 1.0, x_test, y_test
        ),
        'knn': lambda: train_knn_classifier(
            x_train, y_train, 5, x_test, y_test
        ),
        'decision_tree': lambda: train_decision_tree_classifier(
            x_train, y_train, 5, 2, x_test, y_test
        ),
        'random_forest': lambda: train_random_forest_classifier(
            x_train, y_train, 100, None, x_test, y_test
        ),
        'neural_network': lambda: train_neural_network_classifier(
            x_train, y_train, (50, 25), 500, x_test, y_test
        )
    }

    for model_name in models:
        if model_name in model_functions:
            result = model_functions[model_name]()
            results_list.append({
                'model': model_name,
                'train_accuracy': result['train_metrics']['accuracy'],
                'train_f1': result['train_metrics']['f1'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_f1': result['test_metrics']['f1']
            })

    return pd.DataFrame(results_list)


# =============================================================================
# 可視化関数
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    混同行列を可視化する。

    混同行列は、実際のクラスと予測クラスの関係を表す表です。
    対角成分が正解、非対角成分が誤分類を示します。

    Args:
        y_true: 実際のラベル
        y_pred: 予測ラベル
        class_names: クラス名のリスト
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='Actual',
        xlabel='Predicted'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 各セルに数値を表示
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    ROC曲線を可視化する。

    ROC曲線は、閾値を変化させたときの真陽性率と偽陽性率の関係を示します。
    曲線下面積（AUC）が1に近いほど良いモデルです。

    Args:
        y_true: 実際のラベル（二値）
        y_prob: 陽性クラスの予測確率
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc_value:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', label='Random classifier')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
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
    print("分類モデルモジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    x_data, y_data = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    print(f"訓練データ: {x_train.shape[0]}サンプル")
    print(f"テストデータ: {x_test.shape[0]}サンプル")
    print(f"クラス分布: {np.bincount(y_train)}")

    # 各モデルの学習と評価
    print("\n【2. 各モデルの学習と評価】")
    print("-" * 50)

    # ロジスティック回帰
    print("\n[ロジスティック回帰]")
    lr_result = train_logistic_regression(x_train, y_train, 1.0, x_test, y_test)
    print(f"  訓練精度: {lr_result['train_metrics']['accuracy']:.4f}")
    print(f"  テスト精度: {lr_result['test_metrics']['accuracy']:.4f}")

    # SVM
    print("\n[SVM]")
    svm_result = train_svm_classifier(x_train, y_train, 'rbf', 1.0, x_test, y_test)
    print(f"  訓練精度: {svm_result['train_metrics']['accuracy']:.4f}")
    print(f"  テスト精度: {svm_result['test_metrics']['accuracy']:.4f}")

    # ランダムフォレスト
    print("\n[ランダムフォレスト]")
    rf_result = train_random_forest_classifier(
        x_train, y_train, 100, None, x_test, y_test
    )
    print(f"  訓練精度: {rf_result['train_metrics']['accuracy']:.4f}")
    print(f"  テスト精度: {rf_result['test_metrics']['accuracy']:.4f}")

    # モデル比較
    print("\n【3. モデル比較】")
    print("-" * 50)

    comparison_df = compare_classification_models(
        x_train, y_train, x_test, y_test
    )
    print(comparison_df.to_string(index=False))

    # 可視化
    print("\n【4. 可視化】")
    print("-" * 50)

    fig1 = plot_confusion_matrix(
        y_test, rf_result['test_predictions'],
        class_names=['Class 0', 'Class 1'],
        title="Random Forest Confusion Matrix"
    )
    plt.close(fig1)
    print("混同行列: 作成完了")

    fig2 = plot_roc_curve(
        y_test, rf_result['test_probabilities'],
        title="Random Forest ROC Curve"
    )
    plt.close(fig2)
    print("ROC曲線: 作成完了")

    print("\n処理完了!")
