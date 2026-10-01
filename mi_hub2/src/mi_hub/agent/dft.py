"""DFT（VASP）入力の自動生成。

vasp_inputs/ のB2生成パイプラインと同じ規約（INCAR / POSCAR / KPOINTS）で、
エージェントの承認ゲート付きジョブ投入（scheduler.py）に接続するための
最小限の入力生成を提供する。POTCAR は擬ポテンシャルのライセンス上
生成せず、HPC 側の make_potcar.sh / VASP_PP_PATH に委ねる。
"""

from __future__ import annotations

import os

DEFAULT_INCAR = {
    "PREC": "Accurate",
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "ISMEAR": 1,
    "SIGMA": 0.2,
    "IBRION": 2,
    "ISIF": 3,
    "NSW": 60,
    "ISPIN": 2,
    "LREAL": ".FALSE.",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}


def format_incar(overrides: dict[str, object] | None = None) -> str:
    params: dict[str, object] = dict(DEFAULT_INCAR)
    params.update(overrides or {})
    return "\n".join(f"{k} = {v}" for k, v in params.items()) + "\n"


def format_kpoints(mesh: tuple[int, int, int] = (11, 11, 11)) -> str:
    return (
        "Automatic mesh\n0\nGamma\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    )


def format_poscar_b2(elem_a: str, elem_b: str, a0: float) -> str:
    """B2（CsCl型）2原子セルの POSCAR。Aがコーナー、Bが体心。"""
    return (
        f"B2 {elem_a}{elem_b}\n"
        f"{a0:.6f}\n"
        "1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n"
        f"{elem_a} {elem_b}\n1 1\nDirect\n"
        "0.0 0.0 0.0\n0.5 0.5 0.5\n"
    )


def format_poscar(comment: str, a0: float,
                  lattice: list[list[float]],
                  species: list[str], counts: list[int],
                  positions: list[list[float]]) -> str:
    """任意構造（Direct座標）の POSCAR。"""
    if len(species) != len(counts):
        raise ValueError("species と counts の長さが一致しません")
    if sum(counts) != len(positions):
        raise ValueError("counts の合計と positions の数が一致しません")
    lines = [comment, f"{a0:.6f}"]
    lines += [" ".join(f"{x:.10f}" for x in row) for row in lattice]
    lines += [" ".join(species), " ".join(str(c) for c in counts), "Direct"]
    lines += [" ".join(f"{x:.10f}" for x in p) for p in positions]
    return "\n".join(lines) + "\n"


def write_vasp_inputs(workdir: str, poscar: str,
                      incar_overrides: dict[str, object] | None = None,
                      kmesh: tuple[int, int, int] = (11, 11, 11)) -> list[str]:
    """INCAR / POSCAR / KPOINTS を workdir に書き出し、ファイル名一覧を返す。"""
    os.makedirs(workdir, exist_ok=True)
    files = {
        "INCAR": format_incar(incar_overrides),
        "POSCAR": poscar,
        "KPOINTS": format_kpoints(kmesh),
    }
    for name, content in files.items():
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
            f.write(content)
    return sorted(files)
