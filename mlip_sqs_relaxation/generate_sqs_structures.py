#!/usr/bin/env python3
"""Generate binary SQS supercells (BCC/FCC 4x4x4) for MLIP relaxation.

For each unordered pair of the 23 HEA elements:
  - Compositions 0/25/50/75/100 at.% of the second element.
  - Pure endpoints (0, 100 at.%) are generated once as ideal supercells.
  - Mixed compositions (25, 50, 75 at.%) are generated as SQS
    (Special Quasi-random Structures) via icet, with N_SEEDS independent
    random seeds per composition.

Supercell sizes (conventional cubic cells):
  - BCC 4x4x4 -> 128 atoms
  - FCC 4x4x4 -> 256 atoms

Output: extxyz files under <outdir>/<lattice>/<A>-<B>/, plus a manifest CSV.

Usage:
    python generate_sqs_structures.py --lattice bcc --outdir structures
    python generate_sqs_structures.py --lattice fcc --elements "Fe,Ni,Cr"
    python generate_sqs_structures.py --lattice bcc --pairs "Fe-Ni,Nb-Ta"
"""

import argparse
import csv
import itertools
import math
import os

from ase import Atoms
from ase.build import bulk
from ase.io import write as ase_write

HEA_ELEMENTS = sorted([
    'Al', 'Ag', 'Au', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Ir', 'Mn', 'Mo',
    'Nb', 'Ni', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Ta', 'Ti', 'V', 'W', 'Zr',
])

# Metallic (12-coordinate) radii in Angstrom, used only to build a
# reasonable initial lattice constant via Vegard's law; the MLIP
# relaxation optimizes the cell afterwards.
METALLIC_RADIUS = {
    'Ag': 1.445, 'Al': 1.432, 'Au': 1.442, 'Co': 1.251, 'Cr': 1.249,
    'Cu': 1.278, 'Fe': 1.241, 'Hf': 1.580, 'Ir': 1.357, 'Mn': 1.350,
    'Mo': 1.363, 'Nb': 1.429, 'Ni': 1.246, 'Pd': 1.376, 'Pt': 1.387,
    'Re': 1.375, 'Rh': 1.345, 'Ru': 1.339, 'Ta': 1.430, 'Ti': 1.462,
    'V': 1.316, 'W': 1.367, 'Zr': 1.603,
}

COMPOSITIONS = [0, 25, 50, 75, 100]  # at.% of element B
SUPERCELL = 4  # 4x4x4 conventional cells


def lattice_constant(lattice: str, elements, fractions) -> float:
    """Vegard-interpolated cubic lattice constant from metallic radii."""
    r = sum(f * METALLIC_RADIUS[e] for e, f in zip(elements, fractions))
    if lattice == 'bcc':
        return 4.0 * r / math.sqrt(3.0)
    return 2.0 * math.sqrt(2.0) * r


def make_supercell(lattice: str, element: str, a: float) -> Atoms:
    prim = bulk(element, lattice, a=a, cubic=True)
    return prim.repeat((SUPERCELL, SUPERCELL, SUPERCELL))


def generate_pure(lattice: str, element: str) -> Atoms:
    a = lattice_constant(lattice, [element], [1.0])
    return make_supercell(lattice, element, a)


def generate_sqs(lattice: str, elem_a: str, elem_b: str, frac_b: float,
                 seed: int, n_steps: int) -> Atoms:
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    a = lattice_constant(lattice, [elem_a, elem_b], [1.0 - frac_b, frac_b])
    prim = bulk(elem_a, lattice, a=a)
    cutoffs = [1.6 * a, 1.2 * a]  # pair and triplet cutoffs
    cs = ClusterSpace(prim, cutoffs, [[elem_a, elem_b]])
    supercell = make_supercell(lattice, elem_a, a)
    sqs = generate_sqs_from_supercells(
        cluster_space=cs,
        supercells=[supercell],
        target_concentrations={elem_a: 1.0 - frac_b, elem_b: frac_b},
        n_steps=n_steps,
        random_seed=seed,
    )
    return sqs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lattice', choices=['bcc', 'fcc'], required=True)
    parser.add_argument('--outdir', default='structures')
    parser.add_argument('--elements', default=None,
                        help='Comma-separated subset of elements')
    parser.add_argument('--pairs', default=None,
                        help='Comma-separated pairs, e.g. "Fe-Ni,Nb-Ta"')
    parser.add_argument('--n-steps', type=int, default=10000,
                        help='MC steps for SQS annealing')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of independent SQS seeds per mixed composition')
    args = parser.parse_args()

    if args.pairs:
        pairs = []
        for p in args.pairs.split(','):
            a, b = sorted(p.strip().split('-'))
            pairs.append((a, b))
    else:
        elements = (sorted(args.elements.replace(' ', '').split(','))
                    if args.elements else HEA_ELEMENTS)
        for e in elements:
            if e not in METALLIC_RADIUS:
                raise ValueError(f'Unknown element: {e}')
        pairs = list(itertools.combinations(elements, 2))

    outroot = os.path.join(args.outdir, args.lattice)
    os.makedirs(outroot, exist_ok=True)
    manifest_path = os.path.join(outroot, 'manifest.csv')
    rows = []

    pure_done = {}
    for elem_a, elem_b in pairs:
        pair_dir = os.path.join(outroot, f'{elem_a}-{elem_b}')
        os.makedirs(pair_dir, exist_ok=True)
        for comp in COMPOSITIONS:
            frac_b = comp / 100.0
            if comp in (0, 100):
                elem = elem_a if comp == 0 else elem_b
                if elem in pure_done:
                    continue
                atoms = generate_pure(args.lattice, elem)
                fname = os.path.join(outroot, f'pure_{elem}.extxyz')
                ase_write(fname, atoms)
                pure_done[elem] = fname
                rows.append([args.lattice, elem, elem, 0, 0,
                             os.path.relpath(fname, outroot), len(atoms)])
                continue
            for seed in range(1, args.n_seeds + 1):
                atoms = generate_sqs(args.lattice, elem_a, elem_b, frac_b,
                                     seed, args.n_steps)
                fname = os.path.join(
                    pair_dir,
                    f'{elem_a}{100 - comp}{elem_b}{comp}_seed{seed}.extxyz')
                ase_write(fname, atoms)
                rows.append([args.lattice, elem_a, elem_b, comp, seed,
                             os.path.relpath(fname, outroot), len(atoms)])
                print(f'Generated {fname} ({len(atoms)} atoms)')

    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lattice', 'element_a', 'element_b',
                         'at_percent_b', 'seed', 'file', 'n_atoms'])
        writer.writerows(rows)
    print(f'Manifest written to {manifest_path} ({len(rows)} structures)')


if __name__ == '__main__':
    main()
