#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
マテリアルズ・インフォマティクス教科書 - 一括動作確認スクリプト
Materials Informatics Textbook - Batch Verification Script
=============================================================================

【使用方法】
    # 全モジュールの動作確認
    python verify_all.py

    # 特定モジュールのみ確認
    python verify_all.py --module regression_models

    # 詳細出力モード
    python verify_all.py --verbose

    # Materials Project APIキーを指定
    python verify_all.py --api-key YOUR_API_KEY

【対象】
    教員・TA向け動作確認用スクリプト

【出力】
    - 各モジュールの動作確認結果
    - エラーがあった場合の詳細情報
    - 全体のサマリー

=============================================================================
"""

import argparse
import importlib
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 設定
# =============================================================================

# 確認対象モジュール
MODULES_TO_VERIFY = [
    'data_preprocessing',
    'pca_analysis',
    'regression_models',
    'classification_models',
    'clustering_analysis',
    'cross_validation',
    'bayesian_optimization',
    'deep_learning_models',
    'pymatgen_utils',
    'matminer_features',
    'materials_project_api',
]

# オプションモジュール（インストールされていない場合はスキップ）
OPTIONAL_MODULES = [
    'pymatgen_utils',
    'matminer_features',
    'materials_project_api',
]


# =============================================================================
# ユーティリティ関数
# =============================================================================

def print_header(text: str, char: str = '=', width: int = 70) -> None:
    """ヘッダーを表示する"""
    print(char * width)
    print(text.center(width))
    print(char * width)


def print_section(text: str, char: str = '-', width: int = 50) -> None:
    """セクションヘッダーを表示する"""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(char * width)


def print_result(name: str, success: bool, message: str = '') -> None:
    """結果を表示する"""
    status = "[OK]" if success else "[NG]"
    if message:
        print(f"  {status} {name}: {message}")
    else:
        print(f"  {status} {name}")


# =============================================================================
# モジュール検証関数
# =============================================================================

def verify_data_preprocessing(verbose: bool = False) -> Tuple[bool, str]:
    """data_preprocessingモジュールの動作確認"""
    try:
        from mi_textbook import data_preprocessing as dp

        # サンプルデータ
        import pandas as pd
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'a', 'b', 'a']
        })

        # 欠損値処理（関数名: clean_missing_values）
        df_filled = dp.clean_missing_values(df.copy(), strategy='mean')

        # 正規化（関数名: normalize_features）
        df_normalized, _ = dp.normalize_features(df[['A', 'B']].dropna())

        return True, "全機能正常"
    except Exception as e:
        return False, str(e)


def verify_pca_analysis(verbose: bool = False) -> Tuple[bool, str]:
    """pca_analysisモジュールの動作確認"""
    try:
        from mi_textbook import pca_analysis as pca

        # サンプルデータ
        np.random.seed(42)
        x_data = np.random.randn(100, 5)

        # PCA実行（戻り値はタプル: transformed_data, pca_obj, scaler）
        transformed_data, pca_obj, scaler = pca.perform_pca(x_data, n_components=3)

        if transformed_data is None:
            return False, "PCA結果が不正"

        return True, f"次元削減: 5 -> {transformed_data.shape[1]}"
    except Exception as e:
        return False, str(e)


def verify_regression_models(verbose: bool = False) -> Tuple[bool, str]:
    """regression_modelsモジュールの動作確認"""
    try:
        from mi_textbook import regression_models as reg

        # サンプルデータ
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        x_data, y_data = make_regression(n_samples=100, n_features=5, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42
        )

        # 線形回帰
        result = reg.train_linear_regression(x_train, y_train, x_test, y_test)

        if 'test_metrics' not in result:
            return False, "回帰結果が不正"

        r2 = result['test_metrics']['r2']
        return True, f"線形回帰 R²={r2:.4f}"
    except Exception as e:
        return False, str(e)


def verify_classification_models(verbose: bool = False) -> Tuple[bool, str]:
    """classification_modelsモジュールの動作確認"""
    try:
        from mi_textbook import classification_models as clf

        # サンプルデータ
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        x_data, y_data = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42
        )

        # ロジスティック回帰（引数順序: x_train, y_train, c_param, x_test, y_test）
        result = clf.train_logistic_regression(
            x_train, y_train, c_param=1.0, x_test=x_test, y_test=y_test
        )

        if 'test_metrics' not in result:
            return False, "分類結果が不正"

        acc = result['test_metrics']['accuracy']
        return True, f"ロジスティック回帰 Accuracy={acc:.4f}"
    except Exception as e:
        return False, str(e)


def verify_clustering_analysis(verbose: bool = False) -> Tuple[bool, str]:
    """clustering_analysisモジュールの動作確認"""
    try:
        from mi_textbook import clustering_analysis as clust

        # サンプルデータ
        from sklearn.datasets import make_blobs

        x_data, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

        # K-means
        result = clust.train_kmeans(x_data, n_clusters=3)

        if 'labels' not in result:
            return False, "クラスタリング結果が不正"

        # metricsの中にsilhouetteがある
        silhouette = result['metrics']['silhouette']
        return True, f"K-means Silhouette={silhouette:.4f}"
    except Exception as e:
        return False, str(e)


def verify_cross_validation(verbose: bool = False) -> Tuple[bool, str]:
    """cross_validationモジュールの動作確認"""
    try:
        from mi_textbook import cross_validation as cv

        # サンプルデータ
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge

        x_data, y_data = make_regression(n_samples=100, n_features=5, random_state=42)
        model = Ridge()

        # K-fold CV
        result = cv.perform_kfold_cv(model, x_data, y_data, n_splits=5)

        if 'mean_score' not in result:
            return False, "交差検証結果が不正"

        mean_score = result['mean_score']
        return True, f"5-fold CV Mean R²={mean_score:.4f}"
    except Exception as e:
        return False, str(e)


def verify_bayesian_optimization(verbose: bool = False) -> Tuple[bool, str]:
    """bayesian_optimizationモジュールの動作確認"""
    try:
        from mi_textbook import bayesian_optimization as bo

        # サンプルデータ
        np.random.seed(42)
        x_train = np.random.uniform(0, 10, 10).reshape(-1, 1)
        y_train = np.sin(x_train).flatten() + np.random.normal(0, 0.1, 10)

        # GPR
        result = bo.train_gpr(x_train, y_train)

        if 'model' not in result:
            return False, "GPR結果が不正"

        return True, "GPR学習完了"
    except Exception as e:
        return False, str(e)


def verify_deep_learning_models(verbose: bool = False) -> Tuple[bool, str]:
    """deep_learning_modelsモジュールの動作確認"""
    try:
        from mi_textbook import deep_learning_models as dl

        # サンプルデータ
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        x_data, y_data = make_regression(n_samples=100, n_features=5, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42
        )

        # scikit-learn MLP
        result = dl.train_mlp_sklearn(
            x_train, y_train, x_test, y_test,
            hidden_layer_sizes=(32, 16),
            max_iter=100
        )

        if 'test_r2' not in result:
            return False, "MLP結果が不正"

        r2 = result['test_r2']
        return True, f"MLP R²={r2:.4f}"
    except Exception as e:
        return False, str(e)


def verify_pymatgen_utils(verbose: bool = False) -> Tuple[bool, str]:
    """pymatgen_utilsモジュールの動作確認"""
    try:
        from mi_textbook import pymatgen_utils as pmu

        if not pmu.PYMATGEN_AVAILABLE:
            return True, "スキップ（pymatgen未インストール）"

        # FCC構造の作成
        cu_fcc = pmu.create_fcc_structure('Cu', 3.615)
        info = pmu.get_structure_info(cu_fcc)

        if 'formula' not in info:
            return False, "構造情報が不正"

        return True, f"Cu FCC構造作成完了（{info['num_sites']}原子）"
    except Exception as e:
        return False, str(e)


def verify_matminer_features(verbose: bool = False) -> Tuple[bool, str]:
    """matminer_featuresモジュールの動作確認"""
    try:
        from mi_textbook import matminer_features as mmf

        # pymatgenベースの機能をテスト
        if mmf.PYMATGEN_AVAILABLE:
            formulas = ['Li2O', 'Fe2O3', 'TiO2']
            df = mmf.generate_element_statistics(formulas)

            if df.shape[0] != 3:
                return False, "特徴量生成結果が不正"

            return True, f"元素統計量生成完了（{df.shape[1]-1}特徴量）"
        else:
            return True, "スキップ（pymatgen未インストール）"
    except Exception as e:
        return False, str(e)


def verify_materials_project_api(
    verbose: bool = False,
    api_key: Optional[str] = None
) -> Tuple[bool, str]:
    """materials_project_apiモジュールの動作確認"""
    try:
        from mi_textbook import materials_project_api as mpapi

        # APIキーの確認
        if api_key:
            os.environ['MP_API_KEY'] = api_key

        # モジュールのインポート確認のみ
        # 実際のAPI呼び出しは行わない（レート制限対策）
        if hasattr(mpapi, 'get_material_by_id'):
            return True, "モジュールインポート成功（API呼び出しはスキップ）"
        else:
            return False, "必要な関数が見つかりません"
    except Exception as e:
        return False, str(e)


# =============================================================================
# メイン検証関数
# =============================================================================

def run_verification(
    modules: Optional[List[str]] = None,
    verbose: bool = False,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    全モジュールの動作確認を実行する。

    Args:
        modules: 確認するモジュールのリスト（Noneの場合は全モジュール）
        verbose: 詳細出力モード
        api_key: Materials Project APIキー

    Returns:
        検証結果の辞書
    """
    if modules is None:
        modules = MODULES_TO_VERIFY

    # 検証関数のマッピング
    verify_functions = {
        'data_preprocessing': verify_data_preprocessing,
        'pca_analysis': verify_pca_analysis,
        'regression_models': verify_regression_models,
        'classification_models': verify_classification_models,
        'clustering_analysis': verify_clustering_analysis,
        'cross_validation': verify_cross_validation,
        'bayesian_optimization': verify_bayesian_optimization,
        'deep_learning_models': verify_deep_learning_models,
        'pymatgen_utils': verify_pymatgen_utils,
        'matminer_features': verify_matminer_features,
        'materials_project_api': lambda v: verify_materials_project_api(v, api_key),
    }

    results = {
        'total': len(modules),
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'details': {}
    }

    print_header("マテリアルズ・インフォマティクス教科書")
    print_header("一括動作確認", char='-')
    print(f"\n確認対象モジュール数: {len(modules)}")
    print(f"詳細出力: {'ON' if verbose else 'OFF'}")

    start_time = time.time()

    for module_name in modules:
        print_section(f"モジュール: {module_name}")

        if module_name not in verify_functions:
            print_result(module_name, False, "検証関数が未定義")
            results['failed'] += 1
            results['details'][module_name] = {
                'success': False,
                'message': "検証関数が未定義"
            }
            continue

        try:
            success, message = verify_functions[module_name](verbose)

            if 'スキップ' in message:
                results['skipped'] += 1
            elif success:
                results['passed'] += 1
            else:
                results['failed'] += 1

            print_result(module_name, success, message)
            results['details'][module_name] = {
                'success': success,
                'message': message
            }

        except Exception as e:
            results['failed'] += 1
            error_msg = str(e)
            if verbose:
                error_msg = traceback.format_exc()
            print_result(module_name, False, error_msg)
            results['details'][module_name] = {
                'success': False,
                'message': error_msg
            }

    elapsed_time = time.time() - start_time

    # サマリー
    print_header("検証結果サマリー")
    print(f"\n  合計: {results['total']} モジュール")
    print(f"  成功: {results['passed']}")
    print(f"  失敗: {results['failed']}")
    print(f"  スキップ: {results['skipped']}")
    print(f"  実行時間: {elapsed_time:.2f}秒")

    if results['failed'] == 0:
        print("\n  [全モジュール正常動作]")
    else:
        print("\n  [一部モジュールでエラーが発生しました]")
        print("  失敗したモジュール:")
        for name, detail in results['details'].items():
            if not detail['success'] and 'スキップ' not in detail['message']:
                print(f"    - {name}: {detail['message']}")

    return results


# =============================================================================
# コマンドライン引数
# =============================================================================

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='マテリアルズ・インフォマティクス教科書 一括動作確認スクリプト'
    )
    parser.add_argument(
        '--module', '-m',
        type=str,
        help='確認する特定のモジュール名'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細出力モード'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Materials Project APIキー'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='確認対象モジュールの一覧を表示'
    )

    return parser.parse_args()


# =============================================================================
# メイン
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print("確認対象モジュール:")
        for module in MODULES_TO_VERIFY:
            optional = " (オプション)" if module in OPTIONAL_MODULES else ""
            print(f"  - {module}{optional}")
        sys.exit(0)

    modules = [args.module] if args.module else None

    results = run_verification(
        modules=modules,
        verbose=args.verbose,
        api_key=args.api_key
    )

    # 終了コード
    sys.exit(0 if results['failed'] == 0 else 1)
