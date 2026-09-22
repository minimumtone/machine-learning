"""選択された計算コードの入力スクリプト生成。

codes.py の推薦結果から、承認ゲート付きジョブ投入（scheduler.py）へ
そのまま渡せる入力ファイル群と実行スクリプトを組み立てる。
- vasp: dft.py（INCAR/POSCAR/KPOINTS）に委譲
- mlip: CHGNet + ASE の Python スクリプト（ローカル/HPC 両用）
- lammps: EAM/MLIP ポテンシャル用の入力ファイル
- pycalphad: TDB を用いた平衡計算スクリプト
"""

from __future__ import annotations

import os
from typing import Any

from . import dft
from .scheduler import make_sbatch_script

_RUN_COMMANDS = {
    "vasp": "srun vasp_std",
    "mlip": "python3 run_mlip.py",
    "lammps": "srun lmp -in in.lammps",
    "pycalphad": "python3 run_calphad.py",
}


def generate_mlip_script(elements: list[str], structure: str = "fcc",
                         a0: float = 3.6, supercell: int = 2,
                         temperature: float | None = None) -> str:
    """CHGNet + ASE による構造緩和（＋任意で有限温度MD）スクリプト。"""
    md_block = ""
    if temperature:
        md_block = f"""
from ase.md.langevin import Langevin
from ase import units
dyn = Langevin(atoms, timestep=2 * units.fs, temperature_K={temperature},
               friction=0.02)
dyn.run(500)
print("MD後エネルギー [eV]:", atoms.get_potential_energy())
"""
    return f"""# MLIP（CHGNet + ASE）: 構造緩和と全エネルギー
# 注意: MLIP は DFT の代替近似。学習データ外の組成では精度低下に留意。
import random

from ase.build import bulk
from ase.optimize import BFGS
from chgnet.model.dynamics import CHGNetCalculator

elements = {elements!r}
atoms = bulk(elements[0], "{structure}", a={a0}, cubic=True)
atoms = atoms.repeat(({supercell}, {supercell}, {supercell}))
random.seed(42)
for atom in atoms:
    atom.symbol = random.choice(elements)
atoms.calc = CHGNetCalculator()
opt = BFGS(atoms, logfile="relax.log")
opt.run(fmax=0.05, steps=200)
print("原子数:", len(atoms))
print("緩和後エネルギー [eV]:", atoms.get_potential_energy())
print("エネルギー/原子 [eV/atom]:", atoms.get_potential_energy() / len(atoms))
atoms.write("relaxed.cif")
{md_block}"""


def generate_lammps_input(elements: list[str], pair_style: str = "eam/alloy",
                          potential_file: str = "potential.eam.alloy",
                          temperature: float = 300.0,
                          n_steps: int = 10000) -> str:
    """LAMMPS 入力（NPT 平衡化）。構造は data.lammps を別途用意する。"""
    elems = " ".join(elements)
    return f"""# LAMMPS: NPT 平衡化（{elems}）
# 注意: 対象元素系に評価済みのポテンシャルファイルが必要
units metal
boundary p p p
atom_style atomic
read_data data.lammps

pair_style {pair_style}
pair_coeff * * {potential_file} {elems}

thermo 100
thermo_style custom step temp press pe vol
velocity all create {temperature} 42 mom yes rot yes
fix npt1 all npt temp {temperature} {temperature} 0.1 iso 0.0 0.0 1.0
timestep 0.001
dump d1 all custom 1000 dump.lammpstrj id type x y z
run {n_steps}
write_data final.lammps
"""


def generate_pycalphad_script(elements: list[str], tdb_file: str,
                              phases: list[str] | None = None,
                              t_min: float = 300.0, t_max: float = 2000.0) -> str:
    """pycalphad による平衡相計算スクリプト。"""
    return f"""# pycalphad: 平衡相の温度依存性（{'-'.join(elements)}）
# 注意: TDB データベースの評価範囲外の組成・温度は外挿となる
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

plt.rcParams["font.size"] = 20
dbf = Database("{tdb_file}")
comps = {[e.upper() for e in elements]!r} + ["VA"]
phases = {phases!r} or sorted(dbf.phases.keys())
conds = {{v.T: (float({t_min}), float({t_max}), 50.0), v.P: 101325, v.N: 1}}
n_indep = len({elements!r}) - 1
for el in {[e.upper() for e in elements]!r}[:n_indep]:
    conds[v.X(el)] = 1.0 / len({elements!r})
eq = equilibrium(dbf, comps, phases, conds)
print(eq.Phase.values.squeeze())
np.save("equilibrium_phases.npy", eq.Phase.values)
print("平衡計算が完了しました（equilibrium_phases.npy）")
"""


def generate_inputs(code: str, workdir: str, *, elements: list[str],
                    params: dict[str, Any] | None = None) -> dict[str, Any]:
    """コード別の入力一式を workdir に書き出す。

    返り値: {"files": 生成ファイル一覧, "command": 実行コマンド,
             "sbatch": sbatch スクリプト本文}
    """
    p = params or {}
    os.makedirs(workdir, exist_ok=True)
    files: list[str]
    if code == "vasp":
        if len(elements) == 2 and p.get("structure", "b2") == "b2":
            poscar = dft.format_poscar_b2(elements[0], elements[1],
                                          float(p.get("a0", 2.9)))
        else:
            poscar = p.get("poscar") or ""
            if not poscar:
                raise ValueError("vasp: B2 二元系以外は params['poscar'] が必要です")
        files = dft.write_vasp_inputs(workdir, poscar,
                                      incar_overrides=p.get("incar"),
                                      kmesh=tuple(p.get("kmesh", (11, 11, 11))))
    elif code == "mlip":
        script = generate_mlip_script(
            elements, structure=p.get("structure", "fcc"),
            a0=float(p.get("a0", 3.6)), supercell=int(p.get("supercell", 2)),
            temperature=p.get("temperature"))
        _write(workdir, "run_mlip.py", script)
        files = ["run_mlip.py"]
    elif code == "lammps":
        script = generate_lammps_input(
            elements, pair_style=p.get("pair_style", "eam/alloy"),
            potential_file=p.get("potential_file", "potential.eam.alloy"),
            temperature=float(p.get("temperature", 300.0)),
            n_steps=int(p.get("n_steps", 10000)))
        _write(workdir, "in.lammps", script)
        files = ["in.lammps"]
    elif code == "pycalphad":
        if not p.get("tdb_file"):
            raise ValueError("pycalphad: params['tdb_file']（TDBパス）が必要です")
        script = generate_pycalphad_script(
            elements, p["tdb_file"], phases=p.get("phases"),
            t_min=float(p.get("t_min", 300.0)),
            t_max=float(p.get("t_max", 2000.0)))
        _write(workdir, "run_calphad.py", script)
        files = ["run_calphad.py"]
    else:
        raise ValueError(f"未対応の計算コード: {code}")
    command = p.get("command") or _RUN_COMMANDS[code]
    sbatch = make_sbatch_script(
        command, p.get("job_name", f"{code}_{'-'.join(elements).lower()}"),
        partition=p.get("partition"), nodes=int(p.get("nodes", 1)),
        ntasks=int(p.get("ntasks", 1)),
        time_limit=p.get("time_limit", "01:00:00"),
        modules=p.get("modules"))
    return {"files": files, "command": command, "sbatch": sbatch}


def _write(workdir: str, name: str, content: str) -> None:
    with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
        f.write(content)
