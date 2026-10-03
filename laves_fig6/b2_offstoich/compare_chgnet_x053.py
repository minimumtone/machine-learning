#!/usr/bin/env python3
"""CHGNet vs MACE-MP-0 cross-check for B2-NiAl at x_Al = 0.53.

The MACE-relaxed structures (perfect B2, Ni-vacancy and Al-antisite SQS seeds
s0-s2) are re-relaxed with CHGNet using the MACE protocol (LBFGS +
FrechetCellFilter, fmax = 0.02 eV/A, fcc Ni/Al references).  F_vib(T) is
computed for perfect, vacancy s0 and antisite s0 with phonopy (mesh 4x4x4,
traditional FC) with the residual forces of the undisplaced structure
subtracted.  CHGNet F_vib is evaluated at several displacements to check its
convergence (see relax_chgnet/diag_chgnet_disp.py).  The vacancy/antisite Boltzmann weights at
x = 0.53 are evaluated directly at the structure compositions for both
potentials:

    G_i = Ef_i + F_vib,i(T) - kT ln C(n_sub, n_defect) / n_atoms

Outputs:
  analysis/chgnet_x053_structures.csv : per-structure E_form, a_eff, V/atom
  analysis/chgnet_x053_comparison.csv : per-model / per-T G, p_i, c_vac, a_mix
"""
import os, math, time, csv
import numpy as np
import torch
from ase.build import bulk
from ase.io import read, write
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from chgnet.model.dynamics import CHGNetCalculator
from phonopy import Phonopy
from compute_phonon_fvib import phonopy_from_ase, compute_force_sets, fvib_from_phonopy

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, 'analysis')
RELAX = os.path.join(BASE, 'relax')
OUT = os.path.join(BASE, 'relax_chgnet')
KB_EV = 8.617333262e-5
T_LIST = [1273.0, 1473.0]
FMAX = 0.02
NCELL = 64
N_SUB = 64
MACE_DISPLACEMENT = 0.01
CHGNET_DISPLACEMENTS = (0.01, 0.03, 0.05)
MESH = (4, 4, 4)
SEEDS = (0, 1, 2)

torch.set_num_threads(8)
CALC = CHGNetCalculator(use_device='cpu')


def relax(atoms, name):
    atoms.calc = CALC
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    t0 = time.time()
    opt.run(fmax=FMAX, steps=500)
    conv = bool(opt.converged())
    e = float(atoms.get_potential_energy())
    print(f'[relax] {name}: N={len(atoms)} E={e:.4f} conv={conv} ({time.time()-t0:.1f}s)', flush=True)
    write(os.path.join(OUT, name + '.extxyz'), atoms)
    return atoms, e, conv


def phonon_fvib(atoms, name, distance):
    ph = Phonopy(phonopy_from_ase(atoms), supercell_matrix=(1, 1, 1), primitive_matrix='P')
    t0 = time.time()
    ph.forces = compute_force_sets(ph, CALC, distance=distance)
    fvib = fvib_from_phonopy(ph, T_LIST, MESH)
    freqs = ph.mesh.frequencies
    n_imag = int(np.sum(freqs < -0.1))
    print(f'[phonon] {name} d={distance}: {len(ph.displacements)} disp, min f = {freqs.min():.3f} THz, '
          f'n_imag(<-0.1 THz) = {n_imag}/{freqs.size} ({time.time()-t0:.1f}s)', flush=True)
    return fvib, float(freqs.min()), n_imag, int(freqs.size)


def ln_comb(n, k):
    if k <= 0 or k >= n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def read_mace_rows():
    rows = {}
    for fn in ('b2_offstoich_volumes.csv', 'b2_offstoich_volumes_50competition.csv'):
        with open(os.path.join(AN, fn)) as fp:
            for r in csv.DictReader(fp):
                rows.setdefault(r['structure_id'], r)
    return rows


def read_mace_fvib(name):
    with open(os.path.join(RELAX, name + '_fvib.csv')) as fp:
        r = next(csv.DictReader(fp))
    return {T: float(r[f'F_vib_{T:.0f}K']) for T in T_LIST}


def main():
    os.makedirs(OUT, exist_ok=True)
    mu = {}
    for el, a0 in (('Ni', 3.52), ('Al', 4.05)):
        at = bulk(el, 'fcc', a=a0, cubic=True).repeat((2, 2, 2))
        _, e, _ = relax(at, f'pure_{el}_fcc')
        mu[el] = e / len(at)

    names = ['b2_x0.500_perfect'] + [f'b2_x0.530_{b}_s{s}' for b in ('vacNi', 'antisiteAl') for s in SEEDS]
    mace = read_mace_rows()
    struct_rows = []
    relaxed = {}
    for name in names:
        at, e, conv = relax(read(os.path.join(RELAX, name + '.extxyz')), name)
        relaxed[name] = at
        syms = at.get_chemical_symbols()
        n_ni, n_al, n = syms.count('Ni'), syms.count('Al'), len(at)
        v = float(at.get_volume())
        m = mace[name]
        struct_rows.append(dict(
            structure_id=name, branch=m['branch'], n_Ni=n_ni, n_Al=n_al, n_atoms=n,
            x_Al=n_al / n, converged=conv,
            E_form_chgnet=(e - n_ni * mu['Ni'] - n_al * mu['Al']) / n,
            E_form_mace=float(m['E_form_eV_atom']),
            a_eff_chgnet=(v / NCELL) ** (1 / 3), a_eff_mace=float(m['a_eff_A']),
            V_atom_chgnet=v / n, V_atom_mace=float(m['V_per_atom_A3']),
        ))
    with open(os.path.join(AN, 'chgnet_x053_structures.csv'), 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(struct_rows[0].keys()))
        w.writeheader()
        w.writerows(struct_rows)

    fv_chg, imag = {}, {}
    for d in CHGNET_DISPLACEMENTS:
        fv_chg[d] = {}
        for key, name in (('perfect', 'b2_x0.500_perfect'), ('vacancy', 'b2_x0.530_vacNi_s0'),
                          ('antisite', 'b2_x0.530_antisiteAl_s0')):
            fv, fmin, nim, nf = phonon_fvib(relaxed[name], name, d)
            fv_chg[d][key] = fv
            imag[(d, key)] = (fmin, nim, nf)
    fv_mace = {'perfect': read_mace_fvib('b2_x0.500_perfect'),
               'vacancy': read_mace_fvib('b2_x0.530_vacNi_s0'),
               'antisite': read_mace_fvib('b2_x0.530_antisiteAl_s0')}

    def mean(rows, branch, col):
        return float(np.mean([r[col] for r in rows if r['branch'] == branch]))

    perfect = struct_rows[0]
    n_def = {'vacancy': N_SUB - 57, 'antisite': 68 - N_SUB}
    n_at = {'vacancy': 121, 'antisite': 128}
    c_vac = {'vacancy': 1 - 121 / 128, 'antisite': 0.0}
    out = []
    cases = [('mace', MACE_DISPLACEMENT, fv_mace)] + [('chgnet', d, fv_chg[d]) for d in CHGNET_DISPLACEMENTS]
    for model, disp, fv in cases:
        Ef = {b: mean(struct_rows, b, f'E_form_{model}') for b in ('vacancy', 'antisite')}
        a = {b: mean(struct_rows, b, f'a_eff_{model}') for b in ('vacancy', 'antisite')}
        a0 = perfect[f'a_eff_{model}']
        for T in [0.0] + T_LIST:
            kT = KB_EV * T
            G = {}
            for b in ('vacancy', 'antisite'):
                f = fv[b][T] if T > 0 else 0.0
                G[b] = Ef[b] + f - kT * ln_comb(N_SUB, n_def[b]) / n_at[b]
            if T > 0:
                gmin = min(G.values())
                w = {b: math.exp(-(g - gmin) / kT) for b, g in G.items()}
                z = sum(w.values())
                p = {b: w[b] / z for b in w}
            else:
                p = {b: float(G[b] == min(G.values())) for b in G}
            a_mix = sum(p[b] * a[b] for b in p)
            out.append(dict(
                model=model, displacement_A=disp, T_K=T,
                imag_modes_vac=imag[(disp, 'vacancy')][1] if model == 'chgnet' else '',
                imag_modes_anti=imag[(disp, 'antisite')][1] if model == 'chgnet' else '',
                Ef_vac=Ef['vacancy'], Ef_anti=Ef['antisite'],
                Fvib_vac=fv['vacancy'][T] if T > 0 else 0.0,
                Fvib_anti=fv['antisite'][T] if T > 0 else 0.0,
                Fvib_perfect=fv['perfect'][T] if T > 0 else 0.0,
                G_vac=G['vacancy'], G_anti=G['antisite'],
                dG_vac_minus_anti=G['vacancy'] - G['antisite'],
                p_vac=p['vacancy'], p_anti=p['antisite'],
                c_vac_hybrid=sum(p[b] * c_vac[b] for b in p),
                a_perfect=a0, a_vac=a['vacancy'], a_anti=a['antisite'],
                a_mix=a_mix, da_mix_minus_perfect=a_mix - a0,
                da_vac_minus_perfect=a['vacancy'] - a0,
            ))
    with open(os.path.join(AN, 'chgnet_x053_comparison.csv'), 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for k, (fmin, nim, nf) in imag.items():
        print(f'CHGNet {k}: min freq {fmin:.3f} THz, imaginary {nim}/{nf}')
    for r in out:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    main()
