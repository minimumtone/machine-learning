#!/usr/bin/env python3
"""Relax SQS supercells with a machine-learning interatomic potential (MACE).

Reads the manifest produced by generate_sqs_structures.py, relaxes each
structure (atomic positions and cell) by minimizing energy and forces with
MACE-MP-0, and writes relaxed structures plus a results CSV.

Usage:
    python relax_mlip.py --manifest structures/bcc/manifest.csv
    python relax_mlip.py --manifest structures/fcc/manifest.csv \
        --fmax 0.02 --max-steps 500 --device cuda
"""

import argparse
import csv
import os

from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import FIRE


def relax(atoms, calc, fmax: float, max_steps: int, logfile=None):
    atoms.calc = calc
    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=max_steps)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float(abs(forces).max())
    return atoms, energy, max_force, bool(converged), opt.get_number_of_steps()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True,
                        help='manifest.csv from generate_sqs_structures.py')
    parser.add_argument('--outdir', default=None,
                        help='Output dir (default: <manifest dir>/relaxed)')
    parser.add_argument('--fmax', type=float, default=0.02,
                        help='Force convergence criterion (eV/A)')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--model', default='medium',
                        help='MACE-MP-0 model size (small/medium/large)')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only relax the first N structures (for testing)')
    args = parser.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model=args.model, device=args.device,
                   default_dtype='float64', dispersion=False)

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    outdir = args.outdir or os.path.join(manifest_dir, 'relaxed')
    os.makedirs(outdir, exist_ok=True)

    with open(args.manifest, newline='') as f:
        entries = list(csv.DictReader(f))
    if args.limit:
        entries = entries[:args.limit]

    results_path = os.path.join(outdir, 'results.csv')
    fieldnames = ['lattice', 'element_a', 'element_b', 'at_percent_b',
                  'seed', 'n_atoms', 'energy_eV', 'energy_per_atom_eV',
                  'max_force_eV_A', 'volume_A3', 'volume_per_atom_A3',
                  'converged', 'n_opt_steps', 'relaxed_file']
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            src = entry['file']
            if not os.path.isabs(src):
                src = os.path.join(os.path.dirname(manifest_dir) if
                                   os.path.dirname(src) else manifest_dir,
                                   src)
                if not os.path.exists(src):
                    src = entry['file']
            atoms = ase_read(src)
            name = os.path.splitext(os.path.basename(src))[0]
            print(f'Relaxing {name} ({len(atoms)} atoms) ...', flush=True)
            atoms, energy, max_force, converged, n_steps = relax(
                atoms, calc, args.fmax, args.max_steps)
            relaxed_file = os.path.join(outdir, f'{name}_relaxed.extxyz')
            ase_write(relaxed_file, atoms)
            volume = atoms.get_volume()
            n_atoms = len(atoms)
            writer.writerow({
                'lattice': entry['lattice'],
                'element_a': entry['element_a'],
                'element_b': entry['element_b'],
                'at_percent_b': entry['at_percent_b'],
                'seed': entry['seed'],
                'n_atoms': n_atoms,
                'energy_eV': f'{energy:.6f}',
                'energy_per_atom_eV': f'{energy / n_atoms:.6f}',
                'max_force_eV_A': f'{max_force:.6f}',
                'volume_A3': f'{volume:.4f}',
                'volume_per_atom_A3': f'{volume / n_atoms:.4f}',
                'converged': converged,
                'n_opt_steps': n_steps,
                'relaxed_file': relaxed_file,
            })
            f.flush()
            print(f'  E = {energy:.4f} eV ({energy / n_atoms:.4f} eV/atom), '
                  f'fmax = {max_force:.4f} eV/A, '
                  f'converged = {converged} in {n_steps} steps')
    print(f'Results written to {results_path}')


if __name__ == '__main__':
    main()
