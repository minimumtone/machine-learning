#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
matminer特徴量生成モジュール
=============================================================================

【学習目標】
    - matminerライブラリの基本的な使い方を理解する
    - 材料データから機械学習用の特徴量を生成する方法を習得する
    - 組成・構造ベースの特徴量の種類と特性を学ぶ

【前提知識】
    - 材料科学の基礎知識
    - 機械学習の基本概念
    - Python/Pandasの基本操作

【対象】
    材料工学部 3回生

【matminerとは】
    matminerは、材料科学データの特徴量生成に特化したPythonライブラリです。
    組成、結晶構造、電子構造などから、機械学習に適した特徴量を
    自動的に生成できます。

【特徴量の種類】
    - 組成ベース: 元素の物性値の統計量
    - 構造ベース: 結晶構造の幾何学的特徴
    - 電子構造ベース: バンド構造、状態密度の特徴

=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matminerのインポート
try:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    from matminer.featurizers.conversions import StrToComposition
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
    print("警告: matminerがインストールされていません。一部の機能が制限されます。")

# pymatgenのインポート
try:
    from pymatgen.core import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


# =============================================================================
# 組成ベースの特徴量生成
# =============================================================================

def generate_composition_features(
    formulas: List[str],
    feature_set: str = 'magpie'
) -> pd.DataFrame:
    """
    化学組成から特徴量を生成する。

    ElementPropertyフィーチャライザーを使用して、組成から
    元素の物性値に基づく特徴量を生成します。

    【特徴量セット】
    - 'magpie': MAterials-Genome Project Information Exchange
      原子番号、原子量、電気陰性度、価電子数など
    - 'deml': Deml et al.の特徴量セット
    - 'matscholar_el': MatScholarの元素埋め込み

    Args:
        formulas: 化学式のリスト
        feature_set: 使用する特徴量セット

    Returns:
        特徴量を含むDataFrame

    Example:
        >>> formulas = ['Li2O', 'Fe2O3', 'TiO2']
        >>> features = generate_composition_features(formulas)
        >>> print(features.shape)
    """
    if not MATMINER_AVAILABLE:
        raise ImportError("matminerがインストールされていません")

    # DataFrameの作成
    df = pd.DataFrame({'formula': formulas})

    # 文字列からCompositionオブジェクトに変換
    str_to_comp = StrToComposition()
    df = str_to_comp.featurize_dataframe(df, 'formula')

    # ElementProperty特徴量の生成
    ep_featurizer = ElementProperty.from_preset(feature_set)
    df = ep_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)

    return df


def generate_stoichiometry_features(formulas: List[str]) -> pd.DataFrame:
    """
    化学量論的特徴量を生成する。

    組成の化学量論に基づく特徴量を生成します。
    元素数、原子数の比率などが含まれます。

    Args:
        formulas: 化学式のリスト

    Returns:
        特徴量を含むDataFrame
    """
    if not MATMINER_AVAILABLE:
        raise ImportError("matminerがインストールされていません")

    df = pd.DataFrame({'formula': formulas})

    # 文字列からCompositionオブジェクトに変換
    str_to_comp = StrToComposition()
    df = str_to_comp.featurize_dataframe(df, 'formula')

    # Stoichiometry特徴量の生成
    stoich_featurizer = Stoichiometry()
    df = stoich_featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)

    return df


def generate_element_statistics(
    formulas: List[str],
    properties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    元素物性の統計量を特徴量として生成する。

    各組成について、含まれる元素の物性値の統計量
    （平均、標準偏差、最大、最小など）を計算します。

    Args:
        formulas: 化学式のリスト
        properties: 使用する元素物性のリスト

    Returns:
        特徴量を含むDataFrame
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    if properties is None:
        properties = ['X', 'atomic_radius', 'atomic_mass']

    from pymatgen.core import Element

    results = []

    for formula in formulas:
        comp = Composition(formula)
        row = {'formula': formula}

        for prop in properties:
            values = []
            weights = []

            for el, amount in comp.items():
                try:
                    value = getattr(Element(str(el)), prop)
                    if value is not None:
                        values.append(float(value))
                        weights.append(amount)
                except (AttributeError, TypeError):
                    continue

            if values:
                values = np.array(values)
                weights = np.array(weights)
                weights = weights / weights.sum()

                row[f'{prop}_mean'] = np.average(values, weights=weights)
                row[f'{prop}_std'] = np.sqrt(
                    np.average((values - row[f'{prop}_mean'])**2, weights=weights)
                )
                row[f'{prop}_min'] = values.min()
                row[f'{prop}_max'] = values.max()
                row[f'{prop}_range'] = values.max() - values.min()

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# 特徴量選択・前処理
# =============================================================================

def select_features_by_variance(
    df: pd.DataFrame,
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    分散に基づいて特徴量を選択する。

    分散が閾値以下の特徴量（ほぼ定数の特徴量）を除去します。

    Args:
        df: 特徴量DataFrame
        threshold: 分散の閾値

    Returns:
        選択後のDataFrameと選択された特徴量名のタプル
    """
    # 数値列のみを対象
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 分散を計算
    variances = df[numeric_cols].var()

    # 閾値以上の分散を持つ特徴量を選択
    selected_cols = variances[variances > threshold].index.tolist()

    return df[selected_cols], selected_cols


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    相関の高い特徴量を除去する。

    相関係数が閾値以上の特徴量ペアから、一方を除去します。
    多重共線性の問題を軽減できます。

    Args:
        df: 特徴量DataFrame
        threshold: 相関係数の閾値

    Returns:
        選択後のDataFrameと除去された特徴量名のタプル
    """
    # 数値列のみを対象
    numeric_df = df.select_dtypes(include=[np.number])

    # 相関行列を計算
    corr_matrix = numeric_df.corr().abs()

    # 上三角行列を取得
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 閾値以上の相関を持つ特徴量を特定
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    return df.drop(columns=to_drop), to_drop


def normalize_features(
    df: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    特徴量を正規化する。

    Args:
        df: 特徴量DataFrame
        method: 正規化方法 ('standard', 'minmax')

    Returns:
        正規化後のDataFrameとスケーラー情報のタプル
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_normalized = df.copy()

    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    scaler_info = {
        'method': method,
        'columns': numeric_cols,
        'scaler': scaler
    }

    return df_normalized, scaler_info


# =============================================================================
# 特徴量分析
# =============================================================================

def analyze_feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    method: str = 'random_forest'
) -> pd.DataFrame:
    """
    特徴量の重要度を分析する。

    Args:
        features: 特徴量DataFrame
        target: 目的変数
        method: 重要度計算方法 ('random_forest', 'correlation')

    Returns:
        特徴量重要度のDataFrame
    """
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    x_data = features[numeric_cols].fillna(0)

    if method == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_data, target)

        importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    else:  # correlation
        correlations = []
        for col in numeric_cols:
            corr = np.corrcoef(x_data[col], target)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)

        importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': correlations
        }).sort_values('importance', ascending=False)

    return importance_df


# =============================================================================
# 可視化関数
# =============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    特徴量重要度を可視化する。

    Args:
        importance_df: 特徴量重要度のDataFrame
        top_n: 表示する上位特徴量数
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    top_features = importance_df.head(top_n)

    ax.barh(
        range(len(top_features)),
        top_features['importance'],
        alpha=0.7
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_feature_correlation(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    特徴量の相関行列を可視化する。

    Args:
        df: 特徴量DataFrame
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# 便利関数
# =============================================================================

def create_feature_pipeline(
    formulas: List[str],
    target: Optional[pd.Series] = None,
    feature_set: str = 'magpie',
    remove_low_variance: bool = True,
    remove_correlated: bool = True,
    normalize: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    特徴量生成パイプラインを実行する。

    組成から特徴量を生成し、前処理を一括で行います。

    Args:
        formulas: 化学式のリスト
        target: 目的変数（特徴量重要度計算用、オプション）
        feature_set: 使用する特徴量セット
        remove_low_variance: 低分散特徴量を除去するか
        remove_correlated: 高相関特徴量を除去するか
        normalize: 正規化を行うか

    Returns:
        処理済み特徴量DataFrameとパイプライン情報のタプル
    """
    pipeline_info = {}

    # 特徴量生成
    if MATMINER_AVAILABLE:
        df = generate_composition_features(formulas, feature_set)
    else:
        df = generate_element_statistics(formulas)

    pipeline_info['original_features'] = df.shape[1]

    # 低分散特徴量の除去
    if remove_low_variance:
        df, _ = select_features_by_variance(df, threshold=0.01)
        pipeline_info['after_variance_filter'] = df.shape[1]

    # 高相関特徴量の除去
    if remove_correlated:
        df, dropped = remove_correlated_features(df, threshold=0.95)
        pipeline_info['dropped_correlated'] = dropped
        pipeline_info['after_correlation_filter'] = df.shape[1]

    # 正規化
    if normalize:
        df, scaler_info = normalize_features(df, method='standard')
        pipeline_info['scaler'] = scaler_info

    # 特徴量重要度（targetが指定されている場合）
    if target is not None:
        importance_df = analyze_feature_importance(df, target)
        pipeline_info['feature_importance'] = importance_df

    return df, pipeline_info


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("matminer特徴量生成モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータ
    formulas = ['Li2O', 'Fe2O3', 'TiO2', 'Al2O3', 'SiO2', 'MgO', 'CaO', 'ZnO']

    # 元素統計量の生成（pymatgenのみ使用）
    print("\n【1. 元素統計量の生成】")
    print("-" * 50)

    if PYMATGEN_AVAILABLE:
        stats_df = generate_element_statistics(formulas)
        print(f"生成された特徴量数: {stats_df.shape[1] - 1}")
        print(f"サンプル数: {stats_df.shape[0]}")
        print("\n特徴量の例:")
        print(stats_df[['formula', 'X_mean', 'atomic_mass_mean']].head())
    else:
        print("pymatgenがインストールされていないため、スキップ")

    # matminer特徴量の生成
    print("\n【2. matminer特徴量の生成】")
    print("-" * 50)

    if MATMINER_AVAILABLE:
        comp_df = generate_composition_features(formulas, 'magpie')
        print(f"生成された特徴量数: {comp_df.shape[1]}")

        # 化学量論特徴量
        stoich_df = generate_stoichiometry_features(formulas)
        print(f"化学量論特徴量数: {stoich_df.shape[1]}")
    else:
        print("matminerがインストールされていないため、スキップ")

    # 特徴量選択
    print("\n【3. 特徴量選択】")
    print("-" * 50)

    if PYMATGEN_AVAILABLE:
        # 分散フィルタリング
        filtered_df, selected = select_features_by_variance(stats_df, threshold=0.01)
        print(f"分散フィルタリング後: {len(selected)}特徴量")

        # 相関フィルタリング
        uncorr_df, dropped = remove_correlated_features(filtered_df, threshold=0.95)
        print(f"相関フィルタリング後: {uncorr_df.shape[1]}特徴量")
        print(f"除去された特徴量: {dropped}")

    # 可視化
    print("\n【4. 可視化】")
    print("-" * 50)

    if PYMATGEN_AVAILABLE:
        # ダミーの目的変数を作成
        target = pd.Series(np.random.randn(len(formulas)))

        # 特徴量重要度
        importance = analyze_feature_importance(stats_df, target, method='correlation')
        print("特徴量重要度（上位5）:")
        print(importance.head())

        fig = plot_feature_importance(importance, top_n=10)
        plt.close(fig)
        print("\n特徴量重要度プロット: 作成完了")

    print("\n処理完了!")
