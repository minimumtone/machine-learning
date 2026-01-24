#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Materials Project API データ取得モジュール
=============================================================================

【学習目標】
    - Materials Project APIを使用した材料データの取得方法を理解する
    - APIを通じたデータベースアクセスの基本を学ぶ
    - 取得したデータをPandasデータフレームに変換する方法を習得する

【前提知識】
    - Pythonの基本文法（関数、辞書、リスト）
    - HTTP通信の基礎概念
    - JSON形式のデータ構造

【対象】
    材料工学部 3回生

【Materials Projectとは】
    Materials Projectは、米国ローレンス・バークレー国立研究所が運営する
    世界最大級の計算材料データベースです。第一原理計算（DFT計算）に基づく
    材料の構造、エネルギー、バンドギャップなどの物性データが収録されています。

【APIキーの取得方法】
    1. https://materialsproject.org/ にアクセス
    2. アカウントを作成（無料）
    3. Dashboard → API Keys からAPIキーを取得

=============================================================================
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# =============================================================================
# 定数定義
# =============================================================================

# Materials Project APIのエンドポイントURL
DEFAULT_MP_API_URL = "https://api.materialsproject.org/materials/summary"

# OQMD (Open Quantum Materials Database) のAPIエンドポイント
DEFAULT_OQMD_API_URL = "http://oqmd.org/oqmdapi/formationenergy"

# Aflow-lib のAPIエンドポイント
DEFAULT_AFLOW_API_URL = "http://aflowlib.org/API/aflux/"


# =============================================================================
# 基本的なAPI操作関数
# =============================================================================

def get_material_by_id(
    material_id: str,
    api_key: str,
    api_url: str = DEFAULT_MP_API_URL
) -> Optional[Dict[str, Any]]:
    """
    Materials Projectから材料IDを指定してデータを取得する。

    Materials Projectでは、各材料に一意のID（例：mp-149はシリコン）が
    割り当てられています。このIDを使って特定の材料データを取得できます。

    Args:
        material_id: Materials Project ID（例：'mp-149'はSi）
        api_key: Materials Project APIキー
        api_url: APIエンドポイントURL

    Returns:
        材料データを含む辞書、またはエラー時はNone

    Example:
        >>> api_key = "YOUR_API_KEY"
        >>> data = get_material_by_id('mp-149', api_key)
        >>> if data:
        ...     print(f"化学式: {data.get('formula_pretty', 'N/A')}")
    """
    # APIリクエストのヘッダーにAPIキーを設定
    headers = {"X-API-KEY": api_key}

    # GETリクエストを送信
    response = requests.get(f"{api_url}/{material_id}", headers=headers)

    # レスポンスのステータスコードを確認（200は成功）
    if response.status_code == 200:
        return response.json()
    else:
        print(f"エラー: {response.status_code} - {response.text}")
        return None


def parse_json_material_data(json_string: str) -> Dict[str, Any]:
    """
    JSON形式の材料データ文字列をPython辞書に変換する。

    APIから取得したデータはJSON形式で返されます。
    JSONはJavaScript Object Notationの略で、データ交換に広く使われる形式です。

    Args:
        json_string: JSON形式の文字列

    Returns:
        パースされた辞書

    Example:
        >>> json_data = '{"material_id": "mp-149", "formula": "Si"}'
        >>> data = parse_json_material_data(json_data)
        >>> print(data['formula'])
        Si
    """
    return json.loads(json_string)


# =============================================================================
# pymatgen/mp-apiを使用した高度な検索関数
# =============================================================================

def search_materials_by_elements(
    elements: List[str],
    api_key: str,
    fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    指定した元素を含む材料をpymatgen MPResterで検索する。

    pymatgenライブラリのMPResterクラスを使うと、より高度な検索が可能です。
    元素の組み合わせ、物性値の範囲など、様々な条件で材料を検索できます。

    Args:
        elements: 検索する元素のリスト（例：['Li', 'Fe', 'O']）
        api_key: Materials Project APIキー
        fields: 取得するフィールドのリスト

    Returns:
        検索結果を含むPandas DataFrame

    Example:
        >>> df = search_materials_by_elements(['Li', 'O'], 'API_KEY')
        >>> print(df.head())
    """
    from mp_api.client import MPRester

    # デフォルトで取得するフィールドを設定
    if fields is None:
        fields = [
            "material_id",       # 材料ID
            "formula_pretty",    # 化学式（整形済み）
            "energy_above_hull",  # 凸包からのエネルギー（安定性の指標）
            "band_gap",          # バンドギャップ [eV]
            "density",           # 密度 [g/cm³]
            "volume"             # 単位胞体積 [Å³]
        ]

    # MPResterをコンテキストマネージャとして使用
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=elements,
            fields=fields
        )

    # 検索結果をリストに変換
    data_list = []
    for doc in docs:
        row = {}
        for field in fields:
            row[field] = getattr(doc, field, None)
        data_list.append(row)

    return pd.DataFrame(data_list)


def search_oxide_semiconductors(
    api_key: str,
    bandgap_min: float = 0.5,
    bandgap_max: float = 3.5,
    max_results: int = 100
) -> pd.DataFrame:
    """
    酸化物半導体材料を検索する。

    酸化物半導体は、透明導電膜やトランジスタなど様々な用途に使われます。
    バンドギャップの範囲を指定して、目的に合った材料を探索できます。

    【材料工学的背景】
    - バンドギャップ 0.5-1.5 eV: 赤外線検出器、熱電材料
    - バンドギャップ 1.5-3.0 eV: 太陽電池、LED
    - バンドギャップ 3.0-3.5 eV: 紫外線検出器、透明導電膜

    Args:
        api_key: Materials Project APIキー
        bandgap_min: 最小バンドギャップ [eV]
        bandgap_max: 最大バンドギャップ [eV]
        max_results: 最大取得件数

    Returns:
        酸化物半導体データを含むDataFrame
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=["O"],
            band_gap=(bandgap_min, bandgap_max),
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull",
                "density",
                "spacegroup"
            ],
            num_chunks=1,
            chunk_size=max_results
        )

    # 結果をDataFrameに変換
    data_list = []
    for doc in docs[:max_results]:
        spacegroup_symbol = None
        if doc.spacegroup:
            spacegroup_symbol = doc.spacegroup.symbol

        data_list.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "bandgap_ev": doc.band_gap,
            "energy_above_hull_ev": doc.energy_above_hull,
            "density_g_per_cm3": doc.density,
            "spacegroup": spacegroup_symbol
        })

    return pd.DataFrame(data_list)


def search_materials_by_bandgap(
    api_key: str,
    bandgap_min: float = 1.0,
    bandgap_max: float = 5.0,
    elements: Optional[List[str]] = None,
    max_results: int = 100
) -> pd.DataFrame:
    """
    バンドギャップの範囲を指定して材料を検索する。

    バンドギャップは、材料の電気的・光学的性質を決定する重要な物性値です。
    価電子帯の頂上と伝導帯の底のエネルギー差として定義されます。

    【バンドギャップと材料分類】
    - 0 eV: 金属（導体）
    - 0-3 eV: 半導体
    - 3 eV以上: 絶縁体

    Args:
        api_key: Materials Project APIキー
        bandgap_min: 最小バンドギャップ [eV]
        bandgap_max: 最大バンドギャップ [eV]
        elements: フィルタする元素リスト（オプション）
        max_results: 最大取得件数

    Returns:
        検索結果を含むDataFrame
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        search_kwargs = {
            "band_gap": (bandgap_min, bandgap_max),
            "fields": [
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull",
                "density",
                "volume"
            ],
            "num_chunks": 1,
            "chunk_size": max_results
        }

        if elements:
            search_kwargs["elements"] = elements

        docs = mpr.materials.summary.search(**search_kwargs)

    # 結果をDataFrameに変換
    data_list = []
    for doc in docs[:max_results]:
        data_list.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "bandgap_ev": doc.band_gap,
            "energy_above_hull_ev": doc.energy_above_hull,
            "density_g_per_cm3": doc.density,
            "volume_angstrom3": doc.volume
        })

    return pd.DataFrame(data_list)


# =============================================================================
# 他のデータベースからのデータ取得
# =============================================================================

def fetch_oqmd_data(
    elements: Optional[List[str]] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    OQMD (Open Quantum Materials Database) からデータを取得する。

    OQMDは、熱力学的安定性（生成エネルギー）の評価に重点を置いた
    計算材料データベースです。新物質の合成可能性予測や相図構築に有用です。

    【OQMDの特徴】
    - 60万以上の計算データを収録
    - 生成エネルギー、バンドギャップ、磁気モーメントなど
    - APIキー不要で利用可能

    Args:
        elements: フィルタする元素リスト（オプション）
        limit: 最大取得件数

    Returns:
        OQMDデータを含むDataFrame
    """
    params = {
        "limit": limit,
        "format": "json"
    }

    if elements:
        params["filter"] = f"element_set={','.join(elements)}"

    response = requests.get(DEFAULT_OQMD_API_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        if "data" in data:
            return pd.DataFrame(data["data"])
        return pd.DataFrame(data)
    else:
        print(f"エラー: {response.status_code}")
        return pd.DataFrame()


def fetch_aflow_data(
    catalog: str = "icsd",
    elements: Optional[List[str]] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    Aflow-lib データベースからデータを取得する。

    AFLOWは自動計算フレームワークとその大規模データベースです。
    構造・熱力学的安定性に加え、機械特性、輸送特性、格子振動（フォノン）
    など、極めて多様な計算物性データを網羅的に提供します。

    【カタログの種類】
    - icsd: 実験的に決定された結晶構造
    - lib1, lib2, lib3: 理論的に予測された構造

    Args:
        catalog: データベースカタログ ('icsd', 'lib1', 'lib2', 'lib3')
        elements: フィルタする元素リスト（オプション）
        limit: 最大取得件数

    Returns:
        Aflowデータを含むDataFrame
    """
    query = f"?catalog({catalog})"

    if elements:
        species_query = ",".join(elements)
        query += f",species({species_query})"

    query += f",paging(1,{limit})"

    url = DEFAULT_AFLOW_API_URL + query

    response = requests.get(url)

    if response.status_code == 200:
        try:
            data = response.json()
            return pd.DataFrame(data)
        except json.JSONDecodeError:
            print("Aflowレスポンスの解析エラー")
            return pd.DataFrame()
    else:
        print(f"エラー: {response.status_code}")
        return pd.DataFrame()


def collect_mp_data_to_dataframe(
    api_key: str,
    elements: List[str],
    fields: Optional[List[str]] = None,
    max_results: int = 500
) -> pd.DataFrame:
    """
    Materials Projectからデータを収集しDataFrameに変換する。

    機械学習モデルの学習データとして使用するため、
    Materials Projectから大量のデータを効率的に収集します。

    Args:
        api_key: Materials Project APIキー
        elements: 検索する元素のリスト
        fields: 取得するフィールドのリスト
        max_results: 最大取得件数

    Returns:
        材料データを含むDataFrame

    Example:
        >>> df = collect_mp_data_to_dataframe('API_KEY', ['Li', 'Co', 'O'])
        >>> print(f"取得件数: {len(df)}")
    """
    from mp_api.client import MPRester

    if fields is None:
        fields = [
            "material_id",
            "formula_pretty",
            "composition",
            "energy_per_atom",
            "formation_energy_per_atom",
            "energy_above_hull",
            "band_gap",
            "is_metal",
            "density",
            "volume",
            "nsites",
            "spacegroup"
        ]

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=elements,
            fields=fields,
            num_chunks=1,
            chunk_size=max_results
        )

    data_list = []
    for doc in docs[:max_results]:
        row = {"material_id": doc.material_id}

        for field in fields:
            if field == "material_id":
                continue

            value = getattr(doc, field, None)

            if field == "composition" and value is not None:
                row["composition_str"] = str(value)
            elif field == "spacegroup" and value is not None:
                row["spacegroup_symbol"] = value.symbol
                row["spacegroup_number"] = value.number
            else:
                row[field] = value

        data_list.append(row)

    return pd.DataFrame(data_list)


# =============================================================================
# メイン実行部（デモンストレーション）
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Materials Project API データ取得モジュール - デモンストレーション")
    print("=" * 70)

    # サンプルJSONデータの解析デモ
    print("\n【1. JSONデータの解析例】")
    print("-" * 50)

    sample_json = '''
    {
      "material_id": "mp-149",
      "formula": "Si",
      "spacegroup": {
        "symbol": "Fd-3m",
        "number": 227
      },
      "formation_energy_per_atom": -4.67
    }
    '''

    data = parse_json_material_data(sample_json)
    print(f"材料ID: {data['material_id']}")
    print(f"化学式: {data['formula']}")
    print(f"空間群: {data['spacegroup']['symbol']} (No. {data['spacegroup']['number']})")
    print(f"生成エネルギー: {data['formation_energy_per_atom']} eV/atom")

    print("\n【2. 利用可能な関数一覧】")
    print("-" * 50)
    print("- get_material_by_id(): 材料IDでデータ取得")
    print("- search_materials_by_elements(): 元素で検索")
    print("- search_oxide_semiconductors(): 酸化物半導体を検索")
    print("- search_materials_by_bandgap(): バンドギャップで検索")
    print("- fetch_oqmd_data(): OQMDからデータ取得")
    print("- fetch_aflow_data(): Aflowからデータ取得")
    print("- collect_mp_data_to_dataframe(): 大量データ収集")

    print("\n※ 実際にAPIを使用するには、APIキーを設定してください")
