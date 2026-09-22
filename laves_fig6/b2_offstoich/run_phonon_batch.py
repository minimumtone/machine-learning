#!/usr/bin/env python3
"""Batch phonon F_vib computations and collect to analysis/phonon_fvib.csv."""
import os, sys, csv, glob, re
from ase import Atoms
from ase.io import read
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from mace.calculators import mace_mp
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
RELAX = os.path.join(BASE, 'relax')
AN = os.path.join(BASE, 'analysis')
KB = 8.617333262e-5
T_LIST = [1273.0, 1473.0]


def load_atoms(path):
    return read(path)


def phonopy_from_ase(at, supercell_matrix=(1, 1, 1)):
    return PhonopyAtoms(
        symbols=at.get_chemical_symbols(),
        positions=at.get_positions(),
        cell=at.get_cell(),
    ), supercell_matrix


def branch_from_name(name):
    if 'vac' in name:
        return 'vacancy'
    if 'antisite' in name or 'anti' in name:
        return 'antisite'
    return 'perfect'


def x_target_from_name(name):
    m = re.search(r'b2_x([0-9]+\.[0-9]+)', name)
    return float(m.group(1)) if m else None


def run_phonon(path):
    at = load_atoms(path)
    ph_atoms, sc = phonopy_from_ase(at)
    ph = Phonopy(ph_atoms, supercell_matrix=sc, primitive_matrix='P')
    calc = mace_mp(model='medium', default_dtype='float64', device='cpu')
    ph.generate_displacements(distance=0.01, is_plusminus=False, is_diagonal=True)
    supercells = ph.supercells_with_displacements
    n_disp = len(supercells)
    n_atom = len(ph.supercell)
    forces = np.zeros((n_disp, n_atom, 3), dtype=float)
    for i, sc in enumerate(supercells):
        a = Atoms(
            symbols=sc.symbols,
            positions=sc.positions,
            cell=sc.cell,
            pbc=True,
        )
        a.calc = calc
        forces[i] = a.get_forces()
    ph.forces = forces
    ph.produce_force_constants(fc_calculator='traditional')
    ph.run_mesh([4, 4, 4])
    ph.run_thermal_properties(t_min=0, t_max=max(T_LIST) + 100, t_step=10)
    tp = ph.thermal_properties
    temps = tp.temperatures
    free = tp.free_energy
    natom = len(ph.primitive if ph.primitive is not None else ph.supercell)
    f_per_atom = np.array([float(f) * 0.010364272 / natom for f in free])
    return {T: float(np.interp(T, temps, f_per_atom)) for T in T_LIST}


def load_existing(fvib_path):
    with open(fvib_path) as fp:
        reader = csv.DictReader(fp)
        row = next(reader)
    return {T: float(row[f'F_vib_{T:.0f}K']) for T in T_LIST}


if __name__ == '__main__':
    torch.set_num_threads(4)
    # select representative s0 files
    files = [
        'b2_x0.500_perfect.extxyz',
        'b2_x0.520_vacNi_s0.extxyz',
        'b2_x0.520_antisiteAl_s0.extxyz',
        'b2_x0.530_vacNi_s0.extxyz',
        'b2_x0.530_antisiteAl_s0.extxyz',
        'b2_x0.540_vacNi_s0.extxyz',
        'b2_x0.560_vacNi_s0.extxyz',
        'b2_x0.560_antisiteAl_s0.extxyz',
        'b2_x0.580_vacNi_s0.extxyz',
        'b2_x0.580_antisiteAl_s0.extxyz',
    ]
    rows = []
    for f in files:
        path = os.path.join(RELAX, f)
        if not os.path.exists(path):
            print('missing', path)
            continue
        fvib_path = path.replace('.extxyz', '_fvib.csv')
        if os.path.exists(fvib_path):
            fvib = load_existing(fvib_path)
            print('reusing', fvib_path, fvib)
        else:
            fvib = run_phonon(path)
        rows.append({
            'file': f,
            'x_target': x_target_from_name(f),
            'branch': branch_from_name(f),
            'n_atoms': len(read(path)),
            'F_vib_1273': fvib[1273.0],
            'F_vib_1473': fvib[1473.0],
        })
        print(f, fvib)
    out_path = os.path.join(AN, 'phonon_fvib.csv')
    with open(out_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['file','x_target','branch','n_atoms','F_vib_1273','F_vib_1473'])
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', out_path)
