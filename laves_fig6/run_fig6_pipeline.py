#!/usr/bin/env python3
"""Fig.6 reproduction pipeline (Yamanouchi & Miura, Mater. Trans. 59 (2018) 546).

Reproduces the atomic-size (average atomic volume) analysis of the Laves phase
paper for the Nb-Ni-Al system using the pre-trained MACE-MP-0 foundation MLIP
(trained on Materials Project PBE-DFT data).

Outputs (all written in a single pass):
  05_analysis/volumes.csv            per-structure volumes / lattice params
  05_analysis/local_environments.csv Voronoi local volumes per site (C14)
  05_analysis/site_energies.csv      Cr/V site-preference energies
  06_figures/fig6a_ni_al_average_atomic_volume.png
  06_figures/fig6b_c14_nb_nial_average_atomic_volume.png
  06_figures/site_volume_comparison.png
  04_relax/*.extxyz                  relaxed structures
"""
import json
import os
import random
import time

import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from ase.spacegroup import crystal
from scipy.spatial import Voronoi, ConvexHull

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "05_analysis")
FIG = os.path.join(BASE, "06_figures")
RELAX = os.path.join(BASE, "04_relax")
for d in (AN, FIG, RELAX):
    os.makedirs(d, exist_ok=True)

FMAX = 0.02  # eV/A

from icet import ClusterSpace  # noqa: E402
from icet.tools.structure_generation import generate_sqs_from_supercells  # noqa: E402
from mace.calculators import mace_mp  # noqa: E402


def new_calc():
    return mace_mp(model="medium", default_dtype="float64", device="cpu")


CALC = new_calc()


def relax(atoms, name, cell_relax=True, log=True):
    atoms.calc = CALC
    target = FrechetCellFilter(atoms) if cell_relax else atoms
    opt = LBFGS(target, logfile=None)
    t0 = time.time()
    opt.run(fmax=FMAX, steps=500)
    conv = opt.converged()
    e = atoms.get_potential_energy()
    v = atoms.get_volume()
    if log:
        print(f"[relax] {name}: N={len(atoms)} E={e:.4f} eV V/atom={v/len(atoms):.4f} A^3 "
              f"conv={conv} ({time.time()-t0:.1f}s)", flush=True)
    ase_write(os.path.join(RELAX, name + ".extxyz"), atoms)
    return atoms, e, v, conv


rows = []


def record(atoms, e, v, conv, *, structure_id, parent, x_al, supercell, sqs_id, site_def,
           rep=(1, 1, 1)):
    cell = atoms.cell.cellpar()
    n = len(atoms)
    a_unit = cell[0] / rep[0]
    c_unit = cell[2] / rep[2]
    rows.append(dict(
        structure_id=structure_id, parent_structure=parent,
        composition=atoms.get_chemical_formula(), x_Al=x_al, supercell=supercell,
        sqs_id=sqs_id, site_definition=site_def,
        energy_eV=e, energy_per_atom_eV=e / n,
        volume_A3=v, volume_per_atom_A3=v / n,
        a_A=a_unit, c_A=c_unit, c_over_a=c_unit / a_unit,
        cell_a_A=cell[0], cell_c_A=cell[2],
        mlip_model="MACE-MP-0 medium (float64)", dft_reference="MP PBE (training data)",
        temperature_K=0, converged=conv, n_atoms=n,
    ))


# ---------------------------------------------------------------- pure elements
print("=== Phase 1: pure elements ===", flush=True)
pure_vol = {}
pure_energy = {}
pure_specs = [
    ("Ni", "fcc", 3.52), ("Al", "fcc", 4.05), ("Nb", "bcc", 3.30),
    ("Cr", "bcc", 2.88), ("V", "bcc", 3.03),
]
for sym, lat, a0 in pure_specs:
    at = bulk(sym, lat, a=a0, cubic=True)
    at, e, v, conv = relax(at, f"pure_{sym}_{lat}")
    pure_vol[sym] = v / len(at)
    pure_energy[sym] = e / len(at)
    record(at, e, v, conv, structure_id=f"pure_{sym}", parent=f"{lat}-{sym}",
           x_al=np.nan, supercell="1x1x1(cubic)", sqs_id="-", site_def="-")

# ---------------------------------------------------------------- B2 NiAl
print("=== Phase 2: B2-NiAl ===", flush=True)
b2 = crystal(["Ni", "Al"], basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
             spacegroup=221, cellpar=[2.89, 2.89, 2.89, 90, 90, 90])
b2, e_b2, v_b2, conv = relax(b2, "B2_NiAl")
v_b2_atom = v_b2 / len(b2)
record(b2, e_b2, v_b2, conv, structure_id="B2_NiAl", parent="B2-NiAl",
       x_al=0.5, supercell="1x1x1", sqs_id="-", site_def="B2")

# ---------------------------------------------------------------- fcc Ni-Al solid solution
print("=== Phase 3: fcc Ni1-xAlx SQS (icet) ===", flush=True)
rng = random.Random(20260809)
SQS_STEPS = 10000
NSEED = 2

fcc_prim = bulk("Ni", "fcc", a=3.52)
fcc_cs = ClusterSpace(fcc_prim, cutoffs=[7.0, 4.5], chemical_symbols=["Ni", "Al"])
fcc_super = [fcc_prim.repeat((2, 4, 4))]  # 32 atoms

fcc_results = {}  # x -> list of V/atom
for n_al in (0, 2, 4, 6, 8):  # of 32 atoms
    x = n_al / 32.0
    vols = []
    nseed = 1 if n_al == 0 else NSEED
    for s in range(nseed):
        if n_al == 0:
            at = bulk("Ni", "fcc", a=3.52, cubic=True).repeat((2, 2, 2))
            rep = (2, 2, 2)
        else:
            rep = (2, 4, 4)
            random.seed(1000 * n_al + s)
            at = generate_sqs_from_supercells(
                fcc_cs, [sc.copy() for sc in fcc_super], n_steps=SQS_STEPS,
                target_concentrations={"A": {"Ni": 1 - x, "Al": x}})
        name = f"fcc_NiAl_x{x:.4f}_s{s}"
        at, e, v, conv = relax(at, name)
        vols.append(v / len(at))
        record(at, e, v, conv, structure_id=name, parent="fcc-Ni(Al)",
               x_al=x, supercell="32at", sqs_id=f"sqs_s{s}", site_def="fcc SQS (icet)",
               rep=rep)
    fcc_results[x] = vols

xs = np.array(sorted(fcc_results))
means = np.array([np.mean(fcc_results[x]) for x in xs])
stds = np.array([np.std(fcc_results[x], ddof=1) if len(fcc_results[x]) > 1 else 0.0 for x in xs])
b_fit, a_fit = np.polyfit(xs, means, 1)
v_extrap = a_fit + b_fit * 0.5
print(f"linear fit: V = {a_fit:.4f} + {b_fit:.4f} x ; extrap x=0.5 -> {v_extrap:.4f} A^3/atom", flush=True)

# ---------------------------------------------------------------- C14 Nb(Ni,Al)2
print("=== Phase 4: C14-Nb(Ni1-xAlx)2 ===", flush=True)

X6H = 0.8306
Z4F = 0.0630


def make_c14(a=4.81, c=7.85):
    """C14 (MgZn2-type, SG194). Returns atoms with site labels 4f/2a/6h.

    Elements are placeholders (Nb on 4f, Ni on 2a & 6h)."""
    at = crystal(["Nb", "Ni", "Cu"],
                 basis=[(1 / 3, 2 / 3, Z4F), (0, 0, 0), (X6H, 2 * X6H - 1, 0.25)],
                 spacegroup=194, cellpar=[a, a, c, 90, 90, 120])
    sites = []
    for s in at.get_chemical_symbols():
        sites.append({"Nb": "4f", "Ni": "2a", "Cu": "6h"}[s])
    syms = ["Nb" if t == "4f" else "Ni" for t in sites]
    at.set_chemical_symbols(syms)
    at.new_array("site", np.array(sites, dtype="U3"))
    return at


assert sorted(make_c14().arrays["site"]) == ["2a"] * 2 + ["4f"] * 4 + ["6h"] * 6


def assign_c14_sites(atoms, prim):
    """Attach 2a/4f/6h labels to an (unrelaxed) icet SQS supercell of the C14 prim."""
    from scipy.spatial import cKDTree
    nrep = int(round(len(atoms) / len(prim)))
    # brute force: find nearest prim-lattice site label via scaled positions of a repeated prim
    for r in ((1, 1, 1), (2, 2, 1), (2, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)):
        if np.prod(r) == nrep:
            ref = prim.repeat(r)
            if np.allclose(ref.cell.array, atoms.cell.array, atol=1e-6):
                tree = cKDTree(ref.get_positions())
                d, idx = tree.query(atoms.get_positions())
                if d.max() < 1e-5:
                    atoms.set_array("site", ref.arrays["site"][idx])
                    return atoms
    raise RuntimeError("could not assign C14 site labels")


c14_prim = make_c14()
c14_symbols = [["Nb"] if s == "4f" else ["Ni", "Al"] for s in c14_prim.arrays["site"]]
c14_cs = ClusterSpace(c14_prim, cutoffs=[6.0], chemical_symbols=c14_symbols)

c14_results = {}  # x -> list of V/atom
c14_x05_structs = []  # (atoms, e) for x=0.5 primitive, for local analysis / site energies
for x in (0.0, 0.25, 0.5, 0.75, 1.0):
    vols = []
    n_b = 8
    n_al = int(round(x * n_b))
    configs = []
    if n_al in (0, n_b):
        configs.append(("ord", None))
    else:
        # enumerate distinct 2a occupations: k Al on 2a (k=0..min(2,n_al)), rest on 6h
        for k in range(0, min(2, n_al) + 1):
            if n_al - k <= 6:
                configs.append((f"al2a{k}", k))
    for tag, k in configs:
        at = make_c14()
        sites = at.arrays["site"]
        syms = np.array(at.get_chemical_symbols())
        if n_al > 0:
            i2a = np.where(sites == "2a")[0]
            i6h = np.where(sites == "6h")[0]
            if k is None:
                syms[np.concatenate([i2a, i6h])] = "Al"
            else:
                syms[i2a[:k]] = "Al"
                pick = rng.sample(list(i6h), n_al - k)
                syms[pick] = "Al"
        at.set_chemical_symbols(list(syms))
        name = f"c14_x{x:.2f}_{tag}"
        at, e, v, conv = relax(at, name)
        vols.append(v / len(at))
        record(at, e, v, conv, structure_id=name, parent="C14-Nb(Ni,Al)2",
               x_al=x, supercell="1x1x1(12at)", sqs_id=tag,
               site_def=f"A=Nb@4f; B: {tag}")
        if x == 0.5:
            c14_x05_structs.append((name, at.copy(), e))
    c14_results[x] = vols

# icet SQS at x=0.5: 12-atom and 48-atom (2x2x1) cells for size dependence
size_check = []
sqs_x05_small = []
for s in range(NSEED):
    random.seed(9000 + s)
    at = generate_sqs_from_supercells(
        c14_cs, [make_c14()], n_steps=SQS_STEPS,
        target_concentrations={"A": {"Ni": 0.5, "Al": 0.5}})
    at = assign_c14_sites(at, make_c14())
    name = f"c14_x0.50_sqs12_s{s}"
    at, e, v, conv = relax(at, name)
    sqs_x05_small.append(v / len(at))
    c14_x05_structs.append((name, at.copy(), e))
    record(at, e, v, conv, structure_id=name, parent="C14-Nb(Ni,Al)2",
           x_al=0.5, supercell="1x1x1(12at)", sqs_id=f"sqs_s{s}", site_def="B SQS (icet)")
for s in range(NSEED):
    random.seed(9100 + s)
    at = generate_sqs_from_supercells(
        c14_cs, [make_c14().repeat((2, 2, 1))], n_steps=SQS_STEPS,
        target_concentrations={"A": {"Ni": 0.5, "Al": 0.5}})
    at = assign_c14_sites(at, make_c14())
    name = f"c14_x0.50_sqs48_s{s}"
    at, e, v, conv = relax(at, name)
    size_check.append(v / len(at))
    record(at, e, v, conv, structure_id=name, parent="C14-Nb(Ni,Al)2",
           x_al=0.5, supercell="2x2x1(48at)", sqs_id=f"sqs_s{s}", site_def="B SQS (icet)",
           rep=(2, 2, 1))

# ---------------------------------------------------------------- Voronoi local volumes
print("=== Phase 5: Voronoi local volumes (C14 x=0.5) ===", flush=True)


def voronoi_volumes(atoms):
    """Per-atom Voronoi volume with periodic images (3x3x3 replication)."""
    n = len(atoms)
    sup = atoms.repeat((3, 3, 3))
    pos = sup.get_positions()
    # central cell atoms: index offset for repeat is cell (0..26); central is (1,1,1)
    # ase.repeat orders: for each image, all atoms; image order is nested loops z fastest?
    # Safer: find atoms closest to center by recomputing central positions
    central = atoms.get_positions() + atoms.cell.sum(axis=0)
    vor = Voronoi(pos)
    vols = np.full(n, np.nan)
    # map central positions to supercell point indices
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    _, idxs = tree.query(central)
    for i, pi in enumerate(idxs):
        region = vor.regions[vor.point_region[pi]]
        if -1 in region or len(region) == 0:
            continue
        vols[i] = ConvexHull(vor.vertices[region]).volume
    return vols


loc_rows = []
for name, at, e in c14_x05_structs:
    vv = voronoi_volumes(at)
    sites = at.arrays["site"]
    syms = np.array(at.get_chemical_symbols())
    for i in range(len(at)):
        r = (3 * vv[i] / (4 * np.pi)) ** (1 / 3) if np.isfinite(vv[i]) else np.nan
        loc_rows.append(dict(structure_id=name, atom_index=i, element=syms[i],
                             site=sites[i], V_voro_A3=vv[i], r_voro_A=r))
    print(f"[voro] {name}: sum(Vi)={np.nansum(vv):.2f} vs Vcell={at.get_volume():.2f}", flush=True)

# ---------------------------------------------------------------- Cr/V site preference
print("=== Phase 6: Cr/V site-selection energies ===", flush=True)
# reference: lowest-energy x=0.5 primitive NbNiAl
ref_name, ref_at, ref_e = min(c14_x05_structs, key=lambda t: t[2])
print(f"reference C14-NbNiAl: {ref_name} E={ref_e:.4f} eV", flush=True)
site_rows = []


def sub_energy(dopant, target_site, replace_elem):
    at = ref_at.copy()
    sites = at.arrays["site"]
    syms = np.array(at.get_chemical_symbols())
    cand = np.where((sites == target_site) & (syms == replace_elem))[0]
    if len(cand) == 0:
        return None
    i = int(cand[0])
    syms[i] = dopant
    at.set_chemical_symbols(list(syms))
    name = f"c14_NbNiAl_{dopant}@{target_site}({replace_elem})"
    at2, e, v, conv = relax(at, name.replace("@", "_at_").replace("(", "_").replace(")", ""))
    record(at2, e, v, conv, structure_id=name, parent=ref_name, x_al=0.5,
           supercell="1x1x1(12at)", sqs_id="-", site_def=f"{dopant}@{target_site} repl {replace_elem}")
    # E_sub = E(doped) - E(ref) + mu(replaced) - mu(dopant)
    esub = e - ref_e + pure_energy[replace_elem] - pure_energy[dopant]
    return esub


for dop in ("Cr", "V"):
    eA = sub_energy(dop, "4f", "Nb")
    eB2a = sub_energy(dop, "2a", "Ni")
    if eB2a is None:
        eB2a = sub_energy(dop, "2a", "Al")
    eB6h = sub_energy(dop, "6h", "Ni")
    if eB6h is None:
        eB6h = sub_energy(dop, "6h", "Al")
    eB = min(x for x in (eB2a, eB6h) if x is not None)
    site_rows.append(dict(dopant=dop, E_sub_A_eV=eA, E_sub_2a_eV=eB2a, E_sub_6h_eV=eB6h,
                          dE_A_minus_B_eV=eA - eB,
                          preference="A(4f)" if eA - eB < 0 else "B(2a/6h)"))
    print(f"[site] {dop}: E_A={eA:.3f} E_2a={eB2a:.3f} E_6h={eB6h:.3f} dE_A-B={eA-eB:.3f} eV", flush=True)

# Ni/Al 2a-6h site exchange from x=0.5 primitive configs
ex = {n: e for n, _, e in c14_x05_structs}

# ---------------------------------------------------------------- write CSVs
import csv  # noqa: E402

with open(os.path.join(AN, "volumes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(AN, "local_environments.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(loc_rows[0].keys()))
    w.writeheader()
    w.writerows(loc_rows)
with open(os.path.join(AN, "site_energies.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(site_rows[0].keys()))
    w.writeheader()
    w.writerows(site_rows)

# ---------------------------------------------------------------- summary quantities
v_nb, v_ni, v_al = pure_vol["Nb"], pure_vol["Ni"], pure_vol["Al"]
v_pure = (v_nb + v_ni + v_al) / 3
v_weighted_extrap = (v_nb + 2 * v_extrap) / 3
v_weighted_b2 = (v_nb + 2 * v_b2_atom) / 3
c14_mean = {x: float(np.mean(v)) for x, v in c14_results.items()}
c14_std = {x: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for x, v in c14_results.items()}
v_c14_sqs12 = float(np.mean(sqs_x05_small))
v_c14 = v_c14_sqs12  # representative x=0.5 value: SQS mean (ordered configs kept separate)
size_dep = abs(np.mean(size_check) - v_c14_sqs12) / v_c14_sqs12 * 100

summary = dict(
    V_Nb=v_nb, V_Ni=v_ni, V_Al=v_al, V_Cr=pure_vol["Cr"], V_V=pure_vol["V"],
    V_B2_NiAl=v_b2_atom,
    fcc_fit_a=a_fit, fcc_fit_b=b_fit, V_NiAl_extrap_x05=v_extrap,
    dV_NiAl_extrap_minus_B2=v_extrap - v_b2_atom,
    V_C14_x05=v_c14,
    V_C14_x05_ordered_mean=c14_mean[0.5], V_C14_x05_ordered_std=c14_std[0.5],
    V_C14_x05_sqs12=v_c14_sqs12,
    V_C14_x05_sqs12_std=float(np.std(sqs_x05_small, ddof=1)) if len(sqs_x05_small) > 1 else 0.0,
    V_C14_x05_48at=float(np.mean(size_check)),
    V_C14_x05_48at_std=float(np.std(size_check, ddof=1)) if len(size_check) > 1 else 0.0,
    size_dependence_percent=size_dep,
    V_pure=v_pure, V_weighted_extrap=v_weighted_extrap, V_weighted_B2=v_weighted_b2,
    dev_weighted_extrap=abs(v_c14 - v_weighted_extrap),
    dev_weighted_B2=abs(v_c14 - v_weighted_b2),
    dev_pure=abs(v_c14 - v_pure),
    hypothesis_supported=bool(abs(v_c14 - v_weighted_extrap) < abs(v_c14 - v_pure)),
    c14_mean=c14_mean, c14_std=c14_std,
    fcc_xs=xs.tolist(), fcc_means=means.tolist(), fcc_stds=stds.tolist(),
    site_energies=site_rows,
    x05_config_energies=ex,
)
with open(os.path.join(AN, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 20, "figure.autolayout": True})

# Fig 6a
fig, ax = plt.subplots(figsize=(9, 7))
ax.errorbar(xs, means, yerr=stds, fmt="o", ms=10, capsize=5, color="tab:blue",
            label="MLIP fcc-Ni(Al) SQS (icet)")
xf = np.linspace(0, 0.5, 100)
ax.plot(xf, a_fit + b_fit * xf, "--", color="tab:blue", label="Linear fit (Ni-rich)")
ax.plot([0.5], [v_extrap], "s", ms=14, mfc="none", mec="tab:blue", mew=2,
        label=r"Extrapolated $\bar{V}_{\mathrm{Ni(Al)}}(x=0.5)$")
ax.plot([0.5], [v_b2_atom], "^", ms=14, color="tab:red", label="B2-NiAl (MLIP)")
ax.set_xlabel(r"$x_{\mathrm{Al}}$ (atomic fraction)")
ax.set_ylabel(r"Average atomic volume ($\mathrm{\AA}^3$/atom)")
ax.legend(fontsize=14)
ax.set_title("Fig. 6(a) reproduction: Ni–Al", fontsize=20)
fig.savefig(os.path.join(FIG, "fig6a_ni_al_average_atomic_volume.png"), dpi=200)
plt.close(fig)

# Fig 6b
fig, ax = plt.subplots(figsize=(9, 7))
cx = sorted(c14_mean)
cm = [c14_mean[x] for x in cx]
cs = [c14_std[x] for x in cx]
ax.errorbar(cx, cm, yerr=cs, fmt="o-", ms=10, capsize=5, color="tab:green",
            label=r"MLIP C14-Nb(Ni$_{1-x}$Al$_x$)$_2$ (ordered configs)")
ax.plot([0.5], [v_c14_sqs12], "*", ms=18, color="tab:green", mec="k",
        label="12-atom SQS (x=0.5)")
ax.axhline(v_pure, color="gray", ls=":", lw=2,
           label=r"$V_{\mathrm{pure}}=(V_{\mathrm{Nb}}+V_{\mathrm{Ni}}+V_{\mathrm{Al}})/3$")
ax.axhline(v_weighted_extrap, color="tab:blue", ls="--", lw=2,
           label=r"$V_{\mathrm{weighted}}$ (Ni(Al) extrap.)")
ax.axhline(v_weighted_b2, color="tab:red", ls="-.", lw=2,
           label=r"$V_{\mathrm{weighted}}$ (B2-NiAl)")
ax.plot([0.5], [np.mean(size_check)], "d", ms=13, color="tab:purple",
        label="48-atom SQS proxy (x=0.5)")
ax.set_xlabel(r"$x_{\mathrm{Al}}$ in Nb(Ni$_{1-x}$Al$_x$)$_2$")
ax.set_ylabel(r"Average atomic volume ($\mathrm{\AA}^3$/atom)")
ax.legend(fontsize=13)
ax.set_title("Fig. 6(b) reproduction: C14-Nb(Ni,Al)$_2$", fontsize=20)
fig.savefig(os.path.join(FIG, "fig6b_c14_nb_nial_average_atomic_volume.png"), dpi=200)
plt.close(fig)

# site volume comparison
import collections  # noqa: E402
agg = collections.defaultdict(list)
for r in loc_rows:
    if np.isfinite(r["V_voro_A3"]):
        agg[(r["site"], r["element"])].append(r["V_voro_A3"])
keys = sorted(agg)
fig, ax = plt.subplots(figsize=(10, 7))
lbls = [f"{s}\n({el})" for s, el in keys]
vals = [np.mean(agg[k]) for k in keys]
errs = [np.std(agg[k]) for k in keys]
ax.bar(range(len(keys)), vals, yerr=errs, capsize=5,
       color=["tab:orange" if s == "4f" else "tab:blue" if s == "2a" else "tab:green" for s, _ in keys])
ax.set_xticks(range(len(keys)), lbls, fontsize=14)
ax.set_ylabel(r"Voronoi volume ($\mathrm{\AA}^3$)")
ax.set_title("C14 NbNiAl: local Voronoi volume by site", fontsize=20)
fig.savefig(os.path.join(FIG, "site_volume_comparison.png"), dpi=200)
plt.close(fig)

print("DONE", flush=True)
