"""Imaginary modes and F_vib of x=0.53 defect cells with and without residual-force subtraction.

usage: diag_residual_force.py {mace,chgnet} distance
"""
import os, sys, numpy as np, torch
from ase import Atoms
from ase.io import read
from phonopy import Phonopy
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
from compute_phonon_fvib import phonopy_from_ase, compute_force_sets
torch.set_num_threads(8)
model, dist = sys.argv[1], float(sys.argv[2])
if model == 'mace':
    from mace.calculators import mace_mp
    calc = mace_mp(model='medium', default_dtype='float64', device='cpu')
    src = os.path.join(BASE, 'relax')
else:
    from chgnet.model.dynamics import CHGNetCalculator
    calc = CHGNetCalculator(use_device='cpu')
    src = HERE
T = [1273.0, 1473.0]
for name in ('b2_x0.530_vacNi_s0', 'b2_x0.530_antisiteAl_s0'):
    at = read(os.path.join(src, name + '.extxyz'))
    ph = Phonopy(phonopy_from_ase(at), supercell_matrix=(1, 1, 1), primitive_matrix='P')
    forces = compute_force_sets(ph, calc, distance=dist)
    sc = ph.supercell
    ref = Atoms(sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True)
    ref.calc = calc
    f0 = ref.get_forces()
    for label, F in (('raw', forces), ('residual-subtracted', forces - f0[None])):
        ph.forces = F
        ph.produce_force_constants(fc_calculator='traditional')
        ph.run_mesh((4, 4, 4))
        f = ph.mesh.frequencies
        ph.run_thermal_properties(temperatures=T)
        fe = ph.thermal_properties.free_energy * 0.010364272 / len(sc)
        print(f'{model} d={dist} {name} {label}: max|F_res|={np.abs(f0).max():.4f} eV/A, '
              f'min f={f.min():.3f} THz, n_imag(<-0.1)={int((f < -0.1).sum())}/{f.size}, '
              f'Fvib1273={fe[0]:.5f} Fvib1473={fe[1]:.5f} eV/atom', flush=True)
