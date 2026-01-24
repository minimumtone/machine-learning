#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
データ前処理モジュール
=============================================================================

【学習目標】
    - 機械学習における前処理の重要性を理解する
    - 欠損値処理、外れ値除去、正規化などの基本手法を習得する
    - scikit-learnの前処理ツールの使い方を学ぶ

【前提知識】
    - Pythonの基本文法
    - NumPy配列の基本操作
    - Pandas DataFrameの基本操作

【対象】
    材料工学部 3回生

【なぜ前処理が重要か】
    機械学習モデルの性能は、入力データの品質に大きく依存します。
    実験データには欠損値、外れ値、スケールの違いなどが含まれることが多く、
    これらを適切に処理しないとモデルの学習が困難になります。

=============================================================================
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)


# =============================================================================
# 欠損値処理
# =============================================================================

def clean_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    データフレームの欠損値を処理する。

    欠損値（NaN）は機械学習モデルの学習を妨げるため、
    適切な方法で補完または除去する必要があります。

    【補完戦略の選び方】
    - mean（平均値）: 正規分布に近いデータに適する
    - median（中央値）: 外れ値が多いデータに適する
    - mode（最頻値）: カテゴリカルデータに適する
    - drop（削除）: 欠損が少ない場合に適する

    Args:
        df: 入力データフレーム
        strategy: 補完戦略 ('mean', 'median', 'mode', 'drop')
        columns: 処理対象の列（Noneの場合は全列）

    Returns:
        欠損値処理済みのデータフレーム

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
        >>> df_clean = clean_missing_values(df, strategy='mean')
    """
    df_clean = df.copy()

    if columns is None:
        columns = df_clean.columns.tolist()

    for col in columns:
        if col not in df_clean.columns:
            continue

        if strategy == "mean":
            # 平均値で補完（数値列のみ）
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == "median":
            # 中央値で補完（数値列のみ）
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == "mode":
            # 最頻値で補完
            mode_value = df_clean[col].mode()
            if len(mode_value) > 0:
                df_clean[col] = df_clean[col].fillna(mode_value[0])
        elif strategy == "drop":
            # 欠損値を含む行を削除
            df_clean = df_clean.dropna(subset=[col])

    return df_clean


# =============================================================================
# 外れ値処理
# =============================================================================

def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    外れ値を検出して除去する。

    外れ値は、測定誤差や異常なサンプルによって生じることがあります。
    これらを除去することで、モデルの汎化性能が向上する場合があります。

    【外れ値検出手法】
    - IQR法: 四分位範囲を使用（ロバスト）
    - Z-score法: 標準偏差を使用（正規分布を仮定）

    Args:
        df: 入力データフレーム
        columns: 処理対象の列（Noneの場合は数値列全て）
        method: 検出手法 ('iqr', 'zscore')
        threshold: 閾値（IQR法では1.5、Z-score法では3が一般的）

    Returns:
        外れ値除去済みのデータフレーム
    """
    df_clean = df.copy()

    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df_clean.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        if method == "iqr":
            # IQR法による外れ値検出
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]

        elif method == "zscore":
            # Z-score法による外れ値検出
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:
                z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                df_clean = df_clean[z_scores <= threshold]

    return df_clean


# =============================================================================
# スケーリング（正規化・標準化）
# =============================================================================

def normalize_features(
    data: Union[np.ndarray, pd.DataFrame],
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[Union[np.ndarray, pd.DataFrame], MinMaxScaler]:
    """
    特徴量を指定範囲に正規化（Min-Maxスケーリング）する。

    Min-Maxスケーリングは、データを指定した範囲（通常は0-1）に
    線形変換します。外れ値に敏感なため、事前に外れ値処理を推奨します。

    【数式】
    x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min

    Args:
        data: 入力データ（配列またはDataFrame）
        feature_range: 変換後の範囲（デフォルト: 0-1）

    Returns:
        正規化されたデータとスケーラーオブジェクトのタプル

    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> normalized, scaler = normalize_features(data)
        >>> print(normalized)
    """
    scaler = MinMaxScaler(feature_range=feature_range)

    if isinstance(data, pd.DataFrame):
        scaled_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
    else:
        scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


def standardize_features(
    data: Union[np.ndarray, pd.DataFrame]
) -> Tuple[Union[np.ndarray, pd.DataFrame], StandardScaler]:
    """
    特徴量を標準化（Z-scoreスケーリング）する。

    標準化は、データを平均0、標準偏差1に変換します。
    多くの機械学習アルゴリズム（SVM、ニューラルネットワークなど）で
    推奨される前処理手法です。

    【数式】
    x_scaled = (x - mean) / std

    Args:
        data: 入力データ（配列またはDataFrame）

    Returns:
        標準化されたデータとスケーラーオブジェクトのタプル

    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> standardized, scaler = standardize_features(data)
        >>> print(f"平均: {standardized.mean(axis=0)}")  # ほぼ0
        >>> print(f"標準偏差: {standardized.std(axis=0)}")  # ほぼ1
    """
    scaler = StandardScaler()

    if isinstance(data, pd.DataFrame):
        scaled_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
    else:
        scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


# =============================================================================
# カテゴリカルデータのエンコーディング
# =============================================================================

def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "label"
) -> Tuple[pd.DataFrame, dict]:
    """
    カテゴリカル変数を数値に変換する。

    機械学習モデルは数値データを入力として受け取るため、
    カテゴリカル変数（文字列など）を数値に変換する必要があります。

    【エンコーディング手法】
    - label: ラベルエンコーディング（順序あり）
    - onehot: ワンホットエンコーディング（順序なし）

    Args:
        df: 入力データフレーム
        columns: 処理対象の列（Noneの場合はobject型の列全て）
        method: エンコーディング手法 ('label', 'onehot')

    Returns:
        エンコード済みデータフレームとエンコーダー辞書のタプル
    """
    df_encoded = df.copy()
    encoders = {}

    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

    for col in columns:
        if col not in df_encoded.columns:
            continue

        if method == "label":
            # ラベルエンコーディング
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder

        elif method == "onehot":
            # ワンホットエンコーディング
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            encoders[col] = list(dummies.columns)

    return df_encoded, encoders


# =============================================================================
# 統合前処理パイプライン
# =============================================================================

def preprocess_material_data(
    df: pd.DataFrame,
    target_column: str,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    missing_strategy: str = "mean",
    remove_outliers_flag: bool = True,
    scaling_method: str = "standard"
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    材料データの前処理を一括で行う。

    この関数は、材料データに対して以下の前処理を順番に適用します：
    1. 欠損値処理
    2. 外れ値除去（オプション）
    3. カテゴリカル変数のエンコーディング
    4. 数値特徴量のスケーリング

    Args:
        df: 入力データフレーム
        target_column: 目的変数の列名
        numeric_columns: 数値特徴量の列名リスト
        categorical_columns: カテゴリカル特徴量の列名リスト
        missing_strategy: 欠損値補完戦略
        remove_outliers_flag: 外れ値除去を行うかどうか
        scaling_method: スケーリング手法 ('standard', 'minmax')

    Returns:
        前処理済み特徴量、目的変数、前処理情報のタプル
    """
    df_processed = df.copy()
    preprocessing_info = {}

    # 数値列の自動検出
    if numeric_columns is None:
        numeric_columns = df_processed.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)

    # カテゴリカル列の自動検出
    if categorical_columns is None:
        categorical_columns = df_processed.select_dtypes(
            include=['object']
        ).columns.tolist()

    # 1. 欠損値処理
    df_processed = clean_missing_values(
        df_processed,
        strategy=missing_strategy,
        columns=numeric_columns + categorical_columns + [target_column]
    )
    preprocessing_info['missing_strategy'] = missing_strategy

    # 2. 外れ値除去
    if remove_outliers_flag:
        df_processed = remove_outliers(
            df_processed,
            columns=numeric_columns + [target_column]
        )
        preprocessing_info['outliers_removed'] = True

    # 3. カテゴリカル変数のエンコーディング
    if categorical_columns:
        df_processed, encoders = encode_categorical(
            df_processed,
            columns=categorical_columns,
            method="label"
        )
        preprocessing_info['encoders'] = encoders

    # 4. 特徴量とターゲットの分離
    target = df_processed[target_column]
    features = df_processed.drop(columns=[target_column])

    # 5. スケーリング
    numeric_features = features.select_dtypes(include=[np.number])
    if scaling_method == "standard":
        scaled_features, scaler = standardize_features(numeric_features)
    else:
        scaled_features, scaler = normalize_features(numeric_features)

    preprocessing_info['scaler'] = scaler
    preprocessing_info['scaling_method'] = scaling_method

    # スケーリング済み特徴量で置換
    for col in numeric_features.columns:
        features[col] = scaled_features[col].values

    return features, target, preprocessing_info


# =============================================================================
# データ分割
# =============================================================================

def create_train_test_split(
    features: Union[np.ndarray, pd.DataFrame],
    target: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple:
    """
    データを訓練セットとテストセットに分割する。

    機械学習では、モデルの汎化性能を評価するために、
    データを訓練用とテスト用に分割します。

    【分割比率の目安】
    - データ量が多い場合: 80:20 または 70:30
    - データ量が少ない場合: 交差検証を推奨

    Args:
        features: 特徴量データ
        target: 目的変数データ
        test_size: テストデータの割合（0-1）
        random_state: 乱数シード（再現性のため）
        stratify: 層化抽出を行うかどうか（分類問題で推奨）

    Returns:
        (x_train, x_test, y_train, y_test) のタプル

    Example:
        >>> x_train, x_test, y_train, y_test = create_train_test_split(
        ...     features, target, test_size=0.2
        ... )
        >>> print(f"訓練データ: {len(x_train)}, テストデータ: {len(x_test)}")
    """
    stratify_param = target if stratify else None

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("データ前処理モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルデータの作成
    print("\n【1. サンプルデータの作成】")
    print("-" * 50)

    np.random.seed(42)
    sample_data = pd.DataFrame({
        'density': [2.3, 2.5, np.nan, 2.8, 100.0, 2.4, 2.6],  # 外れ値あり
        'bandgap': [1.1, 1.5, 1.3, np.nan, 1.4, 1.2, 1.6],
        'crystal_system': ['cubic', 'cubic', 'tetragonal', 'cubic',
                           'hexagonal', 'cubic', 'tetragonal'],
        'formation_energy': [-0.5, -0.3, -0.4, -0.6, -0.2, -0.5, -0.35]
    })

    print("元のデータ:")
    print(sample_data)
    print(f"\n欠損値の数:\n{sample_data.isnull().sum()}")

    # 欠損値処理
    print("\n【2. 欠損値処理】")
    print("-" * 50)

    df_no_missing = clean_missing_values(sample_data, strategy='mean')
    print("欠損値処理後:")
    print(df_no_missing)

    # 外れ値除去
    print("\n【3. 外れ値除去】")
    print("-" * 50)

    df_no_outliers = remove_outliers(df_no_missing, columns=['density'])
    print(f"外れ値除去前: {len(df_no_missing)}行")
    print(f"外れ値除去後: {len(df_no_outliers)}行")

    # スケーリング
    print("\n【4. 標準化】")
    print("-" * 50)

    numeric_data = df_no_outliers[['density', 'bandgap']].values
    standardized, scaler = standardize_features(numeric_data)
    print(f"標準化後の平均: {standardized.mean(axis=0)}")
    print(f"標準化後の標準偏差: {standardized.std(axis=0)}")

    # カテゴリカルエンコーディング
    print("\n【5. カテゴリカルエンコーディング】")
    print("-" * 50)

    df_encoded, encoders = encode_categorical(
        df_no_outliers,
        columns=['crystal_system'],
        method='label'
    )
    print("エンコード後:")
    print(df_encoded[['crystal_system']].head())

    print("\n処理完了!")
