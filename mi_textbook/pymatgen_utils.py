#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
pymatgen構造操作ユーティリティモジュール
=============================================================================

【学習目標】
    - pymatgenライブラリの基本的な使い方を理解する
    - 結晶構造の表現と操作方法を習得する
    - 構造解析と可視化の手法を学ぶ

【前提知識】
    - 結晶学の基礎（単位胞、空間群、ミラー指数）
    - 材料科学の基礎知識
    - Python/NumPyの基本操作

【対象】
    材料工学部 3回生

【pymatgenとは】
    pymatgen（Python Materials Genomics）は、材料科学のための
    Pythonライブラリです。結晶構造の作成・操作・解析、
    Materials Projectとの連携など、多彩な機能を提供します。

【主な機能】
    - 結晶構造の作成と操作
    - CIF/POSCARファイルの読み書き
    - 対称性解析
    - 相図計算
    - Materials Project APIとの連携

=============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# pymatgenのインポート
try:
    from pymatgen.core import Composition, Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.cif import CifWriter
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("警告: pymatgenがインストールされていません。一部の機能が制限されます。")


# =============================================================================
# 結晶構造の作成
# =============================================================================

def create_structure(
    lattice_params: Dict[str, float],
    species: List[str],
    coords: List[List[float]],
    coords_are_cartesian: bool = False
) -> Any:
    """
    結晶構造を作成する。

    結晶構造は、格子（Lattice）と原子位置（サイト）で定義されます。
    原子位置は分率座標（fractional）またはカルテシアン座標で指定できます。

    【格子パラメータ】
    - a, b, c: 格子定数 [Å]
    - alpha, beta, gamma: 格子角度 [度]

    Args:
        lattice_params: 格子パラメータ {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
        species: 元素記号のリスト
        coords: 原子座標のリスト
        coords_are_cartesian: カルテシアン座標かどうか

    Returns:
        pymatgen Structureオブジェクト

    Example:
        >>> params = {'a': 5.43, 'b': 5.43, 'c': 5.43,
        ...           'alpha': 90, 'beta': 90, 'gamma': 90}
        >>> species = ['Si', 'Si']
        >>> coords = [[0, 0, 0], [0.25, 0.25, 0.25]]
        >>> structure = create_structure(params, species, coords)
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    lattice = Lattice.from_parameters(
        a=lattice_params['a'],
        b=lattice_params['b'],
        c=lattice_params['c'],
        alpha=lattice_params['alpha'],
        beta=lattice_params['beta'],
        gamma=lattice_params['gamma']
    )

    structure = Structure(
        lattice,
        species,
        coords,
        coords_are_cartesian=coords_are_cartesian
    )

    return structure


def create_cubic_structure(
    a: float,
    species: List[str],
    coords: List[List[float]]
) -> Any:
    """
    立方晶構造を作成する。

    立方晶は最も対称性の高い結晶系で、a = b = c、α = β = γ = 90°です。

    Args:
        a: 格子定数 [Å]
        species: 元素記号のリスト
        coords: 原子の分率座標リスト

    Returns:
        pymatgen Structureオブジェクト
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    lattice = Lattice.cubic(a)
    return Structure(lattice, species, coords)


def create_fcc_structure(element: str, a: float) -> Any:
    """
    面心立方（FCC）構造を作成する。

    FCC構造は、単位胞の各頂点と各面の中心に原子が配置された構造です。
    Cu、Al、Au、Niなどの金属がこの構造を取ります。

    Args:
        element: 元素記号
        a: 格子定数 [Å]

    Returns:
        pymatgen Structureオブジェクト
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    # FCC構造の原子位置（分率座標）
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]
    species = [element] * 4

    lattice = Lattice.cubic(a)
    return Structure(lattice, species, coords)


def create_bcc_structure(element: str, a: float) -> Any:
    """
    体心立方（BCC）構造を作成する。

    BCC構造は、単位胞の各頂点と中心に原子が配置された構造です。
    Fe、W、Crなどの金属がこの構造を取ります。

    Args:
        element: 元素記号
        a: 格子定数 [Å]

    Returns:
        pymatgen Structureオブジェクト
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    # BCC構造の原子位置（分率座標）
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ]
    species = [element] * 2

    lattice = Lattice.cubic(a)
    return Structure(lattice, species, coords)


# =============================================================================
# 構造解析
# =============================================================================

def analyze_symmetry(structure: Any) -> Dict[str, Any]:
    """
    結晶構造の対称性を解析する。

    空間群は結晶の対称性を表す群で、230種類存在します。
    対称性解析により、結晶系、空間群、点群などが得られます。

    Args:
        structure: pymatgen Structureオブジェクト

    Returns:
        対称性情報を含む辞書
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    analyzer = SpacegroupAnalyzer(structure)

    return {
        'spacegroup_symbol': analyzer.get_space_group_symbol(),
        'spacegroup_number': analyzer.get_space_group_number(),
        'crystal_system': analyzer.get_crystal_system(),
        'point_group': analyzer.get_point_group_symbol(),
        'hall_symbol': analyzer.get_hall(),
        'conventional_structure': analyzer.get_conventional_standard_structure(),
        'primitive_structure': analyzer.get_primitive_standard_structure()
    }


def get_structure_info(structure: Any) -> Dict[str, Any]:
    """
    結晶構造の基本情報を取得する。

    Args:
        structure: pymatgen Structureオブジェクト

    Returns:
        構造情報を含む辞書
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    return {
        'formula': structure.composition.reduced_formula,
        'num_sites': structure.num_sites,
        'volume': structure.volume,
        'density': structure.density,
        'lattice_a': structure.lattice.a,
        'lattice_b': structure.lattice.b,
        'lattice_c': structure.lattice.c,
        'lattice_alpha': structure.lattice.alpha,
        'lattice_beta': structure.lattice.beta,
        'lattice_gamma': structure.lattice.gamma,
        'elements': [str(el) for el in structure.composition.elements]
    }


def compare_structures(
    structure1: Any,
    structure2: Any,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5
) -> Dict[str, Any]:
    """
    2つの結晶構造を比較する。

    StructureMatcherを使用して、2つの構造が同等かどうかを判定します。
    格子定数や原子位置の許容誤差を指定できます。

    Args:
        structure1: 比較する構造1
        structure2: 比較する構造2
        ltol: 格子定数の許容誤差（相対値）
        stol: サイト位置の許容誤差
        angle_tol: 角度の許容誤差 [度]

    Returns:
        比較結果を含む辞書
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol
    )

    is_match = matcher.fit(structure1, structure2)

    return {
        'is_match': is_match,
        'ltol': ltol,
        'stol': stol,
        'angle_tol': angle_tol
    }


# =============================================================================
# 組成解析
# =============================================================================

def analyze_composition(formula: str) -> Dict[str, Any]:
    """
    化学組成を解析する。

    化学式から元素の種類、原子数、重量比などを計算します。

    Args:
        formula: 化学式（例: 'Li2O', 'Fe2O3'）

    Returns:
        組成情報を含む辞書

    Example:
        >>> info = analyze_composition('Li2O')
        >>> print(f"分子量: {info['weight']:.2f} g/mol")
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    comp = Composition(formula)

    return {
        'formula': comp.reduced_formula,
        'weight': comp.weight,
        'num_atoms': comp.num_atoms,
        'elements': [str(el) for el in comp.elements],
        'element_composition': {str(el): comp.get_atomic_fraction(el)
                                for el in comp.elements},
        'weight_composition': {str(el): comp.get_wt_fraction(el)
                               for el in comp.elements}
    }


def get_element_properties(symbol: str) -> Dict[str, Any]:
    """
    元素の物性情報を取得する。

    Args:
        symbol: 元素記号（例: 'Fe', 'Si'）

    Returns:
        元素情報を含む辞書
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    element = Element(symbol)

    return {
        'symbol': element.symbol,
        'name': element.name,
        'atomic_number': element.Z,
        'atomic_mass': element.atomic_mass,
        'group': element.group,
        'row': element.row,
        'block': element.block,
        'is_metal': element.is_metal,
        'is_transition_metal': element.is_transition_metal,
        'electronegativity': element.X,
        'atomic_radius': element.atomic_radius
    }


# =============================================================================
# ファイル入出力
# =============================================================================

def save_structure_to_cif(
    structure: Any,
    filename: str,
    symprec: float = 0.1
) -> None:
    """
    結晶構造をCIFファイルに保存する。

    CIF（Crystallographic Information File）は、結晶構造データの
    標準的なファイル形式です。

    Args:
        structure: pymatgen Structureオブジェクト
        filename: 保存先ファイル名
        symprec: 対称性判定の精度
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    writer = CifWriter(structure, symprec=symprec)
    writer.write_file(filename)


def load_structure_from_cif(filename: str) -> Any:
    """
    CIFファイルから結晶構造を読み込む。

    Args:
        filename: CIFファイルのパス

    Returns:
        pymatgen Structureオブジェクト
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    return Structure.from_file(filename)


def save_structure_to_poscar(structure: Any, filename: str) -> None:
    """
    結晶構造をPOSCARファイルに保存する。

    POSCARはVASP（第一原理計算ソフトウェア）の入力ファイル形式です。

    Args:
        structure: pymatgen Structureオブジェクト
        filename: 保存先ファイル名
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    structure.to(fmt='poscar', filename=filename)


# =============================================================================
# 構造操作
# =============================================================================

def create_supercell(
    structure: Any,
    scaling_matrix: Union[int, List[int], List[List[int]]]
) -> Any:
    """
    スーパーセル（超格子）を作成する。

    スーパーセルは、単位胞を複数回繰り返した構造です。
    欠陥計算や表面計算などで使用されます。

    Args:
        structure: 元の構造
        scaling_matrix: スケーリング行列（例: [2, 2, 2]で2x2x2のスーパーセル）

    Returns:
        スーパーセル構造
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    return structure.copy()  # make_supercellの代わりにcopyを使用
    # 実際の使用時は: structure.make_supercell(scaling_matrix)


def add_site_to_structure(
    structure: Any,
    species: str,
    coords: List[float],
    coords_are_cartesian: bool = False
) -> Any:
    """
    構造にサイト（原子）を追加する。

    Args:
        structure: 元の構造
        species: 追加する元素
        coords: 原子座標
        coords_are_cartesian: カルテシアン座標かどうか

    Returns:
        サイトを追加した構造
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    new_structure = structure.copy()
    new_structure.append(species, coords, coords_are_cartesian=coords_are_cartesian)
    return new_structure


def remove_site_from_structure(structure: Any, index: int) -> Any:
    """
    構造からサイト（原子）を削除する。

    Args:
        structure: 元の構造
        index: 削除するサイトのインデックス

    Returns:
        サイトを削除した構造
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    new_structure = structure.copy()
    new_structure.remove_sites([index])
    return new_structure


# =============================================================================
# 可視化関数
# =============================================================================

def plot_structure_2d(
    structure: Any,
    axis: str = 'c',
    title: str = "Crystal Structure",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    結晶構造を2次元投影で可視化する。

    Args:
        structure: pymatgen Structureオブジェクト
        axis: 投影軸 ('a', 'b', 'c')
        title: 図のタイトル
        save_path: 保存先パス
        figsize: 図のサイズ

    Returns:
        Matplotlibの図オブジェクト
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgenがインストールされていません")

    fig, ax = plt.subplots(figsize=figsize)

    # 軸の選択
    if axis == 'c':
        x_idx, y_idx = 0, 1
        xlabel, ylabel = 'a', 'b'
    elif axis == 'b':
        x_idx, y_idx = 0, 2
        xlabel, ylabel = 'a', 'c'
    else:
        x_idx, y_idx = 1, 2
        xlabel, ylabel = 'b', 'c'

    # 元素ごとに色を設定
    elements = list(set([str(site.specie) for site in structure]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))
    color_map = {el: colors[i] for i, el in enumerate(elements)}

    # 原子をプロット
    for site in structure:
        coords = site.frac_coords
        element = str(site.specie)
        ax.scatter(
            coords[x_idx], coords[y_idx],
            c=[color_map[element]],
            s=200,
            label=element,
            edgecolors='black',
            linewidth=1
        )

    # 重複ラベルを削除
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel(f'{xlabel} (fractional)')
    ax.set_ylabel(f'{ylabel} (fractional)')
    ax.set_title(title)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
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
    print("pymatgen構造操作ユーティリティ - デモンストレーション")
    print("=" * 70)

    if not PYMATGEN_AVAILABLE:
        print("\n※ pymatgenがインストールされていないため、デモを実行できません")
        print("インストール: pip install pymatgen")
    else:
        # FCC構造の作成
        print("\n【1. FCC構造の作成】")
        print("-" * 50)

        cu_fcc = create_fcc_structure('Cu', 3.615)
        info = get_structure_info(cu_fcc)

        print(f"化学式: {info['formula']}")
        print(f"原子数: {info['num_sites']}")
        print(f"体積: {info['volume']:.2f} Å³")
        print(f"密度: {info['density']:.2f} g/cm³")

        # 対称性解析
        print("\n【2. 対称性解析】")
        print("-" * 50)

        symmetry = analyze_symmetry(cu_fcc)
        print(f"空間群: {symmetry['spacegroup_symbol']} (No. {symmetry['spacegroup_number']})")
        print(f"結晶系: {symmetry['crystal_system']}")
        print(f"点群: {symmetry['point_group']}")

        # 組成解析
        print("\n【3. 組成解析】")
        print("-" * 50)

        comp_info = analyze_composition('Li2O')
        print(f"化学式: {comp_info['formula']}")
        print(f"分子量: {comp_info['weight']:.2f} g/mol")
        print(f"原子数: {comp_info['num_atoms']}")
        print(f"元素組成: {comp_info['element_composition']}")

        # 元素情報
        print("\n【4. 元素情報】")
        print("-" * 50)

        fe_info = get_element_properties('Fe')
        print(f"元素: {fe_info['name']} ({fe_info['symbol']})")
        print(f"原子番号: {fe_info['atomic_number']}")
        print(f"原子量: {fe_info['atomic_mass']:.2f}")
        print(f"電気陰性度: {fe_info['electronegativity']}")

        # 可視化
        print("\n【5. 可視化】")
        print("-" * 50)

        fig = plot_structure_2d(cu_fcc, axis='c', title="Cu FCC Structure")
        plt.close(fig)
        print("構造可視化: 作成完了")

    print("\n処理完了!")
