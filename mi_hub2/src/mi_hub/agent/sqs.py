"""SQS（Special Quasi-random Structure）生成。

icet のクラスタ空間＋焼きなましでランダム合金を模す超格子を生成し、
VASP（POSCAR）/ LAMMPS（data.lammps）/ MLIP（extxyz）の各入力に接続する。
icet は任意依存: 未導入環境では generate_sqs_structure が明示的なエラーを返す。
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms

_PROTOTYPES = {"fcc", "bcc", "hcp"}


def normalize_concentrations(elements: list[str],
                             concentrations: dict[str, float] | None = None,
                             ) -> dict[str, float]:
    """組成を正規化する（未指定は等モル）。合計は1に規格化。"""
    if not elements:
        raise ValueError("elements が空です")
    if concentrations:
        missing = [e for e in elements if e not in concentrations]
        if missing:
            raise ValueError(f"組成が未指定の元素があります: {', '.join(missing)}")
        total = sum(float(concentrations[e]) for e in elements)
        if total <= 0:
            raise ValueError("組成の合計が正ではありません")
        return {e: float(concentrations[e]) / total for e in elements}
    return {e: 1.0 / len(elements) for e in elements}


def generate_sqs_structure(elements: list[str],
                           concentrations: dict[str, float] | None = None,
                           *, prototype: str = "fcc", a0: float = 3.6,
                           max_size: int = 16,
                           cutoffs: list[float] | None = None,
                           n_steps: int = 10000) -> Atoms:
    """icet で SQS 超格子を生成して ASE Atoms を返す。

    max_size は超格子の最大原子数（プリミティブセルの倍数から探索）。
    組成は max_size 以下のセルで実現可能な有理数に丸められる点に注意。
    """
    try:
        from icet import ClusterSpace
        from icet.tools.structure_generation import generate_sqs
    except ImportError as exc:
        raise RuntimeError(
            "icet が見つかりません。`pip install icet`（要 python3-dev）を"
            "実行してください") from exc
    from ase.build import bulk

    if prototype not in _PROTOTYPES:
        raise ValueError(f"未対応のプロトタイプ: {prototype}"
                         f"（対応: {', '.join(sorted(_PROTOTYPES))}）")
    conc = normalize_concentrations(elements, concentrations)
    prim = bulk(elements[0], prototype, a=a0)
    cs = ClusterSpace(prim, cutoffs or [a0 * 1.3, a0 * 0.9],
                      [list(elements)] * len(prim))
    return generate_sqs(cluster_space=cs, max_size=max_size,
                        target_concentrations=conc, n_steps=n_steps)


def atoms_to_poscar(atoms: Atoms) -> str:
    """ASE Atoms を POSCAR 文字列（Direct座標）へ変換する。"""
    from ase.io import write as ase_write

    buf = io.StringIO()
    ase_write(buf, atoms, format="vasp", direct=True)
    return buf.getvalue()


def write_sqs_files(atoms: Atoms, workdir: str,
                    formats: list[str] | None = None) -> list[str]:
    """SQS 構造を各計算コード向けファイルとして書き出す。

    formats: "poscar"（POSCAR）/ "lammps"（data.lammps）/ "xyz"（sqs.extxyz）
    """
    from ase.io import write as ase_write

    os.makedirs(workdir, exist_ok=True)
    written: list[str] = []
    for fmt in formats or ["poscar", "lammps", "xyz"]:
        if fmt == "poscar":
            with open(os.path.join(workdir, "POSCAR"), "w",
                      encoding="utf-8") as f:
                f.write(atoms_to_poscar(atoms))
            written.append("POSCAR")
        elif fmt == "lammps":
            ase_write(os.path.join(workdir, "data.lammps"), atoms,
                      format="lammps-data", masses=True)
            written.append("data.lammps")
        elif fmt == "xyz":
            ase_write(os.path.join(workdir, "sqs.extxyz"), atoms,
                      format="extxyz")
            written.append("sqs.extxyz")
        else:
            raise ValueError(f"未対応の出力形式: {fmt}")
    return written
