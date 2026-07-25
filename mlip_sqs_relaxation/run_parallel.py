#!/usr/bin/env python3
"""Parallel SQS generation + MLIP relaxation for the full HEA binary search.

Designed for a Linux VM with many CPU cores. Example on a 24-core machine:

    mkdir -p /work/sqs_mlip
    python mlip_sqs_relaxation/run_parallel.py \
        --workdir /work/sqs_mlip \
        --lattices bcc fcc \
        --n-steps-sqs 10000 \
        --n-steps-relax 500 \
        --fmax 0.02 \
        --model small \
        --device cpu \
        --workers 24

Outputs:
  <workdir>/
    bcc/
      manifest.csv
      <element>-<element>/Fe75Ni25_seed1.extxyz
      relaxed/
        results.csv
        Fe75Ni25_seed1_relaxed.extxyz
    fcc/
      ...
"""

import argparse
import csv
import itertools
import math
import os
import time
from multiprocessing import get_context

from ase.io import read as ase_read
from ase.io import write as ase_write

HEA_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Ir', 'Mn', 'Mo',
    'Nb', 'Ni', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Ta', 'Ti', 'V', 'W', 'Zr',
])

METALLIC_RADIUS = {
    'Ag': 1.445, 'Al': 1.432, 'Au': 1.442, 'Co': 1.251, 'Cr': 1.249,
    'Cu': 1.278, 'Fe': 1.241, 'Hf': 1.580, 'Ir': 1.357, 'Mn': 1.350,
    'Mo': 1.363, 'Nb': 1.429, 'Ni': 1.246, 'Pd': 1.376, 'Pt': 1.387,
    'Re': 1.375, 'Rh': 1.345, 'Ru': 1.339, 'Ta': 1.430, 'Ti': 1.462,
    'V': 1.316, 'W': 1.367, 'Zr': 1.603,
}

COMPOSITIONS = [0, 25, 50, 75, 100]
SUPERCELL = 4


def lattice_constant(lattice: str, elements, fractions):
    r = sum(f * METALLIC_RADIUS[e] for e, f in zip(elements, fractions))
    if lattice == 'bcc':
        return 4.0 * r / math.sqrt(3.0)
    return 2.0 * math.sqrt(2.0) * r


def make_supercell(lattice: str, element: str, a: float):
    from ase.build import bulk
    prim = bulk(element, lattice, a=a, cubic=True)
    return prim.repeat((SUPERCELL, SUPERCELL, SUPERCELL))


def generate_pure(lattice: str, element: str):
    a = lattice_constant(lattice, [element], [1.0])
    return make_supercell(lattice, element, a)


def generate_sqs(lattice, elem_a, elem_b, frac_b, seed, n_steps):
    from ase.build import bulk
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    a = lattice_constant(lattice, [elem_a, elem_b], [1.0 - frac_b, frac_b])
    prim = bulk(elem_a, lattice, a=a)
    cutoffs = [1.6 * a, 1.2 * a]
    cs = ClusterSpace(prim, cutoffs, [[elem_a, elem_b]])
    supercell = make_supercell(lattice, elem_a, a)
    return generate_sqs_from_supercells(
        cluster_space=cs,
        supercells=[supercell],
        target_concentrations={elem_a: 1.0 - frac_b, elem_b: frac_b},
        n_steps=n_steps,
        random_seed=seed,
    )


def generate_one(lattice, elem_a, elem_b, comp, seed, outroot, n_steps):
    frac_b = comp / 100.0
    if comp == 0 or comp == 100:
        elem = elem_a if comp == 0 else elem_b
        atoms = generate_pure(lattice, elem)
        name = f'pure_{elem}'
    else:
        atoms = generate_sqs(lattice, elem_a, elem_b, frac_b, seed, n_steps)
        name = f'{elem_a}{100 - comp}{elem_b}{comp}_seed{seed}'

    pair_dir = os.path.join(outroot, f'{elem_a}-{elem_b}')
    os.makedirs(pair_dir, exist_ok=True)
    fname = os.path.join(pair_dir, f'{name}.extxyz')
    ase_write(fname, atoms)
    return {
        'lattice': lattice,
        'element_a': elem_a,
        'element_b': elem_b,
        'at_percent_b': comp,
        'seed': seed,
        'file': os.path.relpath(fname, outroot),
        'n_atoms': len(atoms),
    }


def all_tasks(lattice, elements, outroot, n_steps, n_seeds):
    tasks = []
    pairs = list(itertools.combinations(elements, 2))
    for elem_a, elem_b in pairs:
        # mixed compositions: 25/50/75, n_seeds each
        for comp in (25, 50, 75):
            for seed in range(1, n_seeds + 1):
                tasks.append((lattice, elem_a, elem_b, comp, seed,
                              outroot, n_steps))
    # pure endpoints once per element
    for elem in elements:
        tasks.append((lattice, elem, elem, 0, 0, outroot, n_steps))
    return tasks


def write_manifest(outroot, rows):
    manifest_path = os.path.join(outroot, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'lattice', 'element_a', 'element_b', 'at_percent_b',
            'seed', 'file', 'n_atoms'])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def generate_lattice(lattice, elements, workdir, n_steps, n_seeds, workers):
    outroot = os.path.join(workdir, lattice)
    os.makedirs(outroot, exist_ok=True)
    tasks = all_tasks(lattice, elements, outroot, n_steps, n_seeds)

    ctx = get_context('spawn')
    with ctx.Pool(workers) as pool:
        rows = pool.starmap(generate_one, tasks)

    manifest_path = write_manifest(outroot, rows)
    print(f'[{lattice}] Generated {len(rows)} structures in {outroot}')
    return manifest_path


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

_worker_calc = None


def _init_worker(model, device, default_dtype):
    global _worker_calc
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from mace.calculators import mace_mp
    _worker_calc = mace_mp(
        model=model, device=device,
        default_dtype=default_dtype, dispersion=False)


def _relax_one(args):
    manifest_dir, entry, outdir, fmax, max_steps = args

    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    base = {
        'lattice': entry['lattice'],
        'element_a': entry['element_a'],
        'element_b': entry['element_b'],
        'at_percent_b': entry['at_percent_b'],
        'seed': entry['seed'],
        'n_atoms': entry['n_atoms'],
        'energy_eV': '',
        'energy_per_atom_eV': '',
        'max_force_eV_A': '',
        'volume_A3': '',
        'volume_per_atom_A3': '',
        'converged': '',
        'n_opt_steps': '',
        'relaxed_file': '',
    }

    src = entry['file']
    if not os.path.isabs(src):
        src = os.path.join(manifest_dir, src)
    try:
        atoms = ase_read(src)
    except Exception as e:
        return {**base, 'error': f'read failed: {e}'}

    atoms.calc = _worker_calc
    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf, logfile=None)
    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        return {**base, 'error': f'opt failed: {e}'}

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float(abs(forces).max())
    volume = atoms.get_volume()
    n_atoms = len(atoms)

    name = os.path.splitext(os.path.basename(src))[0]
    relaxed_file = os.path.join(outdir, f'{name}_relaxed.extxyz')
    ase_write(relaxed_file, atoms)

    return {
        **base,
        'n_atoms': n_atoms,
        'energy_eV': f'{energy:.6f}',
        'energy_per_atom_eV': f'{energy / n_atoms:.6f}',
        'max_force_eV_A': f'{max_force:.6f}',
        'volume_A3': f'{volume:.4f}',
        'volume_per_atom_A3': f'{volume / n_atoms:.4f}',
        'converged': bool(converged),
        'n_opt_steps': opt.get_number_of_steps(),
        'relaxed_file': relaxed_file,
        'error': '',
    }


def relax_lattice(manifest_path, workers, model, device,
                  default_dtype, fmax, max_steps):
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    outdir = os.path.join(manifest_dir, 'relaxed')
    os.makedirs(outdir, exist_ok=True)

    with open(manifest_path, newline='') as f:
        entries = list(csv.DictReader(f))

    args_list = [(manifest_dir, e, outdir, fmax, max_steps) for e in entries]

    fieldnames = ['lattice', 'element_a', 'element_b', 'at_percent_b',
                  'seed', 'n_atoms', 'energy_eV', 'energy_per_atom_eV',
                  'max_force_eV_A', 'volume_A3', 'volume_per_atom_A3',
                  'converged', 'n_opt_steps', 'relaxed_file', 'error']
    results_path = os.path.join(outdir, 'results.csv')

    ctx = get_context('spawn')
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(model, device, default_dtype),
    ) as pool, open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        n = len(args_list)
        for i, result in enumerate(pool.imap_unordered(_relax_one, args_list), 1):
            writer.writerow(result)
            f.flush()
            pair = f"{result['element_a']}-{result['element_b']}"
            err = result.get('error', '')
            if err:
                print(f'[{i}/{n}] ERROR {pair} {result["at_percent_b"]}% '
                      f'seed {result["seed"]}: {err}', flush=True)
            else:
                print(f'[{i}/{n}] relaxed {pair} {result["at_percent_b"]}% '
                      f'seed {result["seed"]}: conv={result["converged"]} '
                      f'steps={result["n_opt_steps"]} '
                      f'energy={result["energy_per_atom_eV"]}', flush=True)

    print(f'Results written to {results_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', default='/tmp/sqs_mlip')
    parser.add_argument('--lattices', nargs='+', default=['bcc', 'fcc'])
    parser.add_argument('--elements', nargs='+', default=HEA_ELEMENTS)
    parser.add_argument('--n-steps-sqs', type=int, default=10000)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--n-steps-relax', type=int, default=500)
    parser.add_argument('--fmax', type=float, default=0.02)
    parser.add_argument('--model', default='small')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--default-dtype', default='float64')
    parser.add_argument('--workers', type=int, default=min(24, os.cpu_count()))
    parser.add_argument('--skip-generate', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    t0 = time.time()
    for lattice in args.lattices:
        if not args.skip_generate:
            manifest = generate_lattice(
                lattice, args.elements, args.workdir,
                args.n_steps_sqs, args.n_seeds, args.workers)
        else:
            manifest = os.path.join(args.workdir, lattice, 'manifest.csv')

        relax_lattice(
            manifest, args.workers,
            args.model, args.device, args.default_dtype,
            args.fmax, args.n_steps_relax)

    elapsed = time.time() - t0
    print(f'All done in {elapsed / 3600:.1f} h')


if __name__ == '__main__':
    main()
