#!/usr/bin/env python3
"""Compute vibrational free energy for an ASE extxyz structure using phonopy + MACE-MP-0."""
import argparse
import time
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from mace.calculators import mace_mp
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

KB_EV = 8.617333262145e-5  # eV/K


def load_atoms(path):
    return read(path)


def phonopy_from_ase(at):
    return PhonopyAtoms(
        symbols=at.get_chemical_symbols(),
        positions=at.get_positions(),
        cell=at.get_cell(),
    )


def compute_force_sets(ph, calc, plusminus=False, distance=0.01):
    """Generate displacements, compute MACE forces for each, return force array."""
    ph.generate_displacements(distance=distance, is_plusminus=plusminus, is_diagonal=True)
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
    return forces


def fvib_from_phonopy(ph, T_list, mesh_dim):
    """Run mesh, thermal properties, return F_vib per atom at each T."""
    ph.produce_force_constants(fc_calculator='traditional')
    ph.run_mesh(mesh_dim)
    ph.run_thermal_properties(t_min=0, t_max=max(T_list) + 100, t_step=10)
    tp = ph.thermal_properties
    temps = tp.temperatures
    free = tp.free_energy  # per unit cell? Phonopy: kJ/mol for primitive cell?
    # Phonopy returns total free energy (kJ/mol) for the primitive cell.
    # We want per atom in the calculation cell (supercell/unitcell used as P).
    natom = len(ph.primitive if ph.primitive is not None else ph.supercell)
    # Convert kJ/mol to eV per atom (1 kJ/mol = 0.01036427 eV per formula unit).
    f_per_atom = np.array([float(f) * 0.010364272 / natom for f in free])
    return {T: float(np.interp(T, temps, f_per_atom)) for T in T_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('structure', help='path to extxyz')
    parser.add_argument('--distance', type=float, default=0.01)
    parser.add_argument('--mesh', type=int, nargs=3, default=[4, 4, 4])
    parser.add_argument('--plusminus', action='store_true')
    parser.add_argument('--T', type=float, nargs='+', default=[1273, 1473])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    at = load_atoms(args.structure)
    pa = phonopy_from_ase(at)
    # Detect symmetry automatically; pass primitive='P' to keep input cell as primitive.
    ph = Phonopy(pa, supercell_matrix=(1, 1, 1), primitive_matrix='P')
    calc = mace_mp(model='medium', default_dtype='float64', device='cpu')
    t0 = time.time()
    forces = compute_force_sets(ph, calc, plusminus=args.plusminus, distance=args.distance)
    tf = time.time()
    print(f'Force calculations: {len(forces)} disp, {tf - t0:.1f} s')
    ph.forces = forces
    f_vib = fvib_from_phonopy(ph, args.T, args.mesh)
    n_atom = len(ph.supercell)
    res = {
        'structure': args.structure,
        'n_atoms': n_atom,
        'formula': at.get_chemical_formula(),
        'volume': float(at.get_volume()),
        'volume_per_atom': float(at.get_volume() / n_atom),
    }
    print(f'F_vib (eV/atom):')
    for T in args.T:
        res[f'F_vib_{T:.0f}K'] = f_vib[T]
        print(f'  T={T}K: {f_vib[T]:.6f} eV/atom')
    # Write CSV row
    out = args.structure.replace('.extxyz', '_fvib.csv')
    import pandas as pd
    pd.DataFrame([res]).to_csv(out, index=False)
    print('Wrote', out)


if __name__ == '__main__':
    main()
