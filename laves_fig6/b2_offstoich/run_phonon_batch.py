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
DISPLACEMENT = 0.01
MESH = (4, 4, 4)
MACE_MODEL = 'medium'
CACHE_VERSION = '1'


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
    calc = mace_mp(model=MACE_MODEL, default_dtype='float64', device='cpu')
    ph.generate_displacements(distance=DISPLACEMENT, is_plusminus=False, is_diagonal=True)
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
    ph.run_mesh(list(MESH))
    ph.run_thermal_properties(t_min=0, t_max=max(T_LIST) + 100, t_step=10)
    tp = ph.thermal_properties
    temps = tp.temperatures
    free = tp.free_energy
    natom = len(ph.primitive if ph.primitive is not None else ph.supercell)
    f_per_atom = np.array([float(f) * 0.010364272 / natom for f in free])
    fvib = {T: float(np.interp(T, temps, f_per_atom)) for T in T_LIST}

    # Cache with source-file metadata so a re-relaxed structure invalidates old results.
    src_stat = os.stat(path)
    row = {
        'structure': path,
        'n_atoms': natom,
        'source_mtime': str(src_stat.st_mtime),
        'source_size': str(src_stat.st_size),
        'distance': str(DISPLACEMENT),
        'mesh': 'x'.join(str(m) for m in MESH),
        'model': MACE_MODEL,
        'cache_version': CACHE_VERSION,
    }
    for T in T_LIST:
        row[f'F_vib_{T:.0f}K'] = fvib[T]
    out_path = path.replace('.extxyz', '_fvib.csv')
    with open(out_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return fvib


def _fvib_key(row, T):
    """Return a value from a cache row, accepting both old and new key names."""
    for suffix in (f'{T:.0f}K', f'{T:.0f}'):
        key = f'F_vib_{suffix}'
        if key in row:
            return row[key]
    raise KeyError(f'No F_vib column for T={T} in cache row')


def load_existing(fvib_path, src_path):
    """Return cached F_vib if the source file has not changed.

    Recompute when the source extxyz is newer than the cache (mtime).  If the
    cache carries metadata (source_size, distance, mesh, model), also validate
    those; legacy caches without metadata are accepted as-is when the source is
    not newer.
    """
    if not os.path.exists(fvib_path):
        return None
    src_stat = os.stat(src_path)
    cache_stat = os.stat(fvib_path)
    if src_stat.st_mtime > cache_stat.st_mtime:
        return None
    with open(fvib_path) as fp:
        reader = csv.DictReader(fp)
        row = next(reader)
    if 'source_size' in row and str(src_stat.st_size) != row['source_size']:
        return None
    if (row.get('distance', str(DISPLACEMENT)) != str(DISPLACEMENT) or
            row.get('mesh', 'x'.join(str(m) for m in MESH)) != 'x'.join(str(m) for m in MESH) or
            row.get('model', MACE_MODEL) != MACE_MODEL):
        return None
    return {T: float(_fvib_key(row, T)) for T in T_LIST}


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
        fvib = load_existing(fvib_path, path)
        if fvib is None:
            fvib = run_phonon(path)
            print('computed', fvib_path, fvib)
        else:
            print('reusing', fvib_path, fvib)
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
