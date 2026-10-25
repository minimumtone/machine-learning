#!/usr/bin/env python3
"""
VASP input generator for dilute-Al solid solution calculations
(Miura 2018 dilute-extrapolation comparison scheme).

Method A (composition series):
  FCC hosts (2x2x2 conventional, 32 atoms): n_Al = 0..4  (0, 3.125, ..., 12.5 at%)
  BCC hosts (3x3x3 conventional, 54 atoms): n_Al = 0..4  (0, 1.85, ..., 7.41 at%)
Method B (single-impurity finite-size check):
  FCC hosts: 3x3x3 conventional (108 atoms), n_Al = 1
  BCC hosts: 4x4x4 conventional (128 atoms), n_Al = 1

Al atoms are placed at maximally separated sites (greedy max-min periodic
distance) to minimize Al-Al interaction within the series.

All settings match generate_all_b2.py (PBE, ENCUT=520, ISIF=3, ISPIN=2,
ISMEAR=1, POTCAR variants incl. W_sv) and the run script uses
mpirun -np $NP $VASPBIN as in generate_magnetic_b2_recalc.py.

Directory structure:
    DILUTE_AL/
    ├── fcc_Ni_n0/ ... fcc_Ni_n4/   (32-atom series)
    ├── fcc_Ni_imp108/              (108-atom single impurity)
    ├── bcc_Nb_n0/ ... bcc_Nb_n4/   (54-atom series)
    ├── bcc_Nb_imp128/              (128-atom single impurity)
    ├── make_potcar.sh
    ├── run_all.sh
    └── extract_results.py

Usage:
    python generate_dilute_al.py
"""

import os
import itertools

import numpy as np

# =====================================================================
# Hosts
#   FCC group 1 (Miura Table 4): Ni, Co, Pd, Rh, Ir
#   FCC group 2 (positive-deviation systems): Cu, Ag, Au, Pt
#   BCC: Nb
# =====================================================================
FCC_HOSTS = {
    # exp/near-exp fcc lattice constants (A) for initial POSCAR
    "Ni": 3.52, "Co": 3.54, "Pd": 3.89, "Rh": 3.80, "Ir": 3.84,
    "Cu": 3.61, "Ag": 4.09, "Au": 4.08, "Pt": 3.92,
}
BCC_HOSTS = {
    "Nb": 3.301,
}

N_AL_SERIES = [0, 1, 2, 3, 4]

# POTCAR variants — identical to generate_all_b2.py
POTCAR_VARIANTS = {
    "Sc": "Sc_sv", "Ti": "Ti_pv", "V": "V_sv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co",   "Ni": "Ni_pv",
    "Cu": "Cu_pv", "Zn": "Zn",
    "Y": "Y_sv",   "Zr": "Zr_sv", "Nb": "Nb_pv", "Mo": "Mo_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd",    "Ag": "Ag",
    "La": "La",    "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_sv",
    "Re": "Re_pv", "Os": "Os_pv", "Ir": "Ir",    "Pt": "Pt",
    "Au": "Au",
    "Ca": "Ca_sv", "Mg": "Mg_pv", "Be": "Be",
    "Al": "Al",    "Si": "Si",    "Ge": "Ge_d",
    "Sn": "Sn_d",  "Pb": "Pb_d",
}

FCC_BASIS = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
             (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
BCC_BASIS = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]


def build_sites(basis, nrep):
    sites = []
    for i, j, k in itertools.product(range(nrep), repeat=3):
        for b in basis:
            sites.append(((i + b[0]) / nrep,
                          (j + b[1]) / nrep,
                          (k + b[2]) / nrep))
    return np.array(sites)


def min_image_dist(f1, f2):
    d = np.abs(f1 - f2)
    d = np.minimum(d, 1.0 - d)
    return np.sqrt((d ** 2).sum())


def pick_al_sites(sites, n_al):
    """Greedy max-min periodic separation (fractional metric; cubic cell)."""
    if n_al == 0:
        return []
    chosen = [0]
    while len(chosen) < n_al:
        best, best_d = None, -1.0
        for c in range(len(sites)):
            if c in chosen:
                continue
            dmin = min(min_image_dist(sites[c], sites[j]) for j in chosen)
            if dmin > best_d:
                best, best_d = c, dmin
        chosen.append(best)
    return sorted(chosen)


def write_incar(dirpath, natoms):
    ncore = 4 if natoms <= 64 else 8
    content = f"""\
SYSTEM = dilute Al solid solution

# Electronic relaxation
ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = .FALSE.

# Ionic relaxation
IBRION = 2
ISIF   = 3
NSW    = 100
EDIFFG = -0.01

# Smearing (metals)
ISMEAR = 1
SIGMA  = 0.2

# Exchange-correlation
GGA    = PE

# Spin polarization (important for 3d metals)
ISPIN  = 2

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance
NCORE  = {ncore}
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_poscar(dirpath, host, a0, nrep, basis, al_idx):
    sites = build_sites(basis, nrep)
    L = a0 * nrep
    natoms = len(sites)
    n_al = len(al_idx)
    n_host = natoms - n_al
    al_set = set(al_idx)
    host_lines = [sites[i] for i in range(natoms) if i not in al_set]
    al_lines = [sites[i] for i in al_idx]

    lines = [f"{host}{n_host}Al{n_al} dilute solid solution", "1.0",
             f"  {L:.6f}  0.000000  0.000000",
             f"  0.000000  {L:.6f}  0.000000",
             f"  0.000000  0.000000  {L:.6f}"]
    if n_al > 0:
        lines += [f"  {host}  Al", f"  {n_host}  {n_al}"]
    else:
        lines += [f"  {host}", f"  {n_host}"]
    lines.append("Direct")
    for s in host_lines:
        lines.append(f"  {s[0]:.6f}  {s[1]:.6f}  {s[2]:.6f}")
    for s in al_lines:
        lines.append(f"  {s[0]:.6f}  {s[1]:.6f}  {s[2]:.6f}")
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return natoms, n_al


def write_kpoints(dirpath, a0, nrep):
    # match B2 density: 16 mesh for a ~ 3 A  ->  k ~ 48 / L
    L = a0 * nrep
    k = max(2, int(round(48.0 / L)))
    content = f"""\
Automatic mesh
0
Gamma
  {k} {k} {k}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


def generate_potcar_script(base_dir, calcs):
    lines = [
        "#!/bin/bash",
        "# POTCAR generation for dilute-Al calculations",
        "# Requires: $VASP_PP_PATH (PAW-PBE directory)",
        "",
        'if [ -z "$VASP_PP_PATH" ]; then',
        '    echo "Error: VASP_PP_PATH is not set."; exit 1',
        'fi',
        "",
        "FAIL=0",
    ]
    for name, host, n_al in calcs:
        pot_host = POTCAR_VARIANTS.get(host, host)
        if n_al > 0:
            cmd = (f'cat "$VASP_PP_PATH"/{pot_host}/POTCAR '
                   f'"$VASP_PP_PATH"/Al/POTCAR > {name}/POTCAR 2>/dev/null')
        else:
            cmd = (f'cat "$VASP_PP_PATH"/{pot_host}/POTCAR '
                   f'> {name}/POTCAR 2>/dev/null')
        lines.append(cmd)
        lines.append(f'if [ $? -ne 0 ]; then echo "  FAIL: {name}"; '
                     'FAIL=$((FAIL+1)); fi')
    lines += ["", f'echo "Done. Failed: $FAIL / {len(calcs)}"', ""]
    p = os.path.join(base_dir, "make_potcar.sh")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    os.chmod(p, 0o755)


def generate_run_script(base_dir, calcs):
    lines = [
        "#!/bin/bash",
        "# Batch execution for dilute-Al calculations",
        "# Runs: mpirun -np $NP $VASPBIN (default NP=8)",
        "",
        'if [ -z "$VASPBIN" ]; then',
        '    echo "Error: VASPBIN is not set."; exit 1',
        'fi',
        'NP="${NP:-8}"',
        "",
        'BASE=$(cd "$(dirname "$0")" && pwd)',
        'LOG="$BASE/run_status.log"',
        'echo "=== Dilute-Al calculations ===" | tee "$LOG"',
        f'echo "Total: {len(calcs)}" | tee -a "$LOG"',
        'echo "Started: $(date)" | tee -a "$LOG"',
        "",
    ]
    for i, (name, _, _) in enumerate(calcs, 1):
        lines += [
            f'echo "[{i}/{len(calcs)}] {name}..." | tee -a "$LOG"',
            f'cd "$BASE/{name}"',
            'if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
            '    echo "  SKIP (already converged)" | tee -a "$LOG"',
            'elif [ ! -f POTCAR ]; then',
            '    echo "  SKIP (no POTCAR)" | tee -a "$LOG"',
            'else',
            '    mpirun -np $NP $VASPBIN > vasp.out 2>&1',
            '    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
            '        echo "  CONVERGED" | tee -a "$LOG"',
            '    else',
            '        echo "  WARNING: not converged" | tee -a "$LOG"',
            '    fi',
            'fi',
            'cd "$BASE"',
            "",
        ]
    lines += ['echo "Finished: $(date)" | tee -a "$LOG"', ""]
    p = os.path.join(base_dir, "run_all.sh")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    os.chmod(p, 0o755)


def generate_extract_script(base_dir):
    content = '''\
#!/usr/bin/env python3
"""Extract volumes/energies from completed dilute-Al VASP calculations."""

import os
import csv
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_contcar(path):
    with open(path) as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    vecs = [[float(x) for x in lines[i].split()] for i in (2, 3, 4)]
    v = [[c * scale for c in row] for row in vecs]
    vol = (v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1])
           - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0])
           + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]))
    counts = [int(x) for x in lines[6].split()]
    return abs(vol), sum(counts)


def read_energy(path):
    with open(path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "F=" in line:
            parts = line.split()
            return float(parts[parts.index("F=") + 1])
    return None


def main():
    rows, not_run = [], []
    for d in sorted(os.listdir(BASE_DIR)):
        dp = os.path.join(BASE_DIR, d)
        if not os.path.isdir(dp):
            continue
        m = re.fullmatch(r"(fcc|bcc)_([A-Z][a-z]?)_(n(\\d+)|imp\\d+)", d)
        if not m:
            continue
        contcar = os.path.join(dp, "CONTCAR")
        if not os.path.exists(contcar) or os.path.getsize(contcar) == 0:
            not_run.append(d)
            continue
        vol, natoms = read_contcar(contcar)
        with open(os.path.join(dp, "POSCAR")) as f:
            pl = f.readlines()
        cnts = [int(x) for x in pl[6].split()]
        els = pl[5].split()
        n_al = cnts[els.index("Al")] if "Al" in els else 0
        osz = os.path.join(dp, "OSZICAR")
        e = read_energy(osz) if os.path.exists(osz) else None
        conv = "yes" if os.path.exists(os.path.join(dp, "OUTCAR")) and \\
            "reached required accuracy" in open(
                os.path.join(dp, "OUTCAR"), errors="ignore").read() else "no"
        rows.append(dict(
            directory=d, lattice=m.group(1), host=m.group(2),
            n_Al=n_al, natoms=natoms, c_Al=n_al / natoms,
            volume_A3=vol, volume_per_atom_A3=vol / natoms,
            energy_eV=e, converged=conv))

    out = os.path.join(BASE_DIR, "dilute_al_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["directory"])
        w.writeheader()
        w.writerows(rows)
    print(f"Extracted {len(rows)} results -> {out}")
    if not_run:
        print(f"Not yet run: {len(not_run)}")


if __name__ == "__main__":
    main()
'''
    p = os.path.join(base_dir, "extract_results.py")
    with open(p, "w") as f:
        f.write(content)
    os.chmod(p, 0o755)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "DILUTE_AL")
    os.makedirs(base_dir, exist_ok=True)
    calcs = []

    def add(name, host, a0, nrep, basis, n_al):
        dp = os.path.join(base_dir, name)
        os.makedirs(dp, exist_ok=True)
        sites = build_sites(basis, nrep)
        al_idx = pick_al_sites(sites, n_al)
        natoms, _ = write_poscar(dp, host, a0, nrep, basis, al_idx)
        write_incar(dp, natoms)
        write_kpoints(dp, a0, nrep)
        calcs.append((name, host, n_al))
        return natoms

    for host, a0 in FCC_HOSTS.items():
        for n in N_AL_SERIES:
            add(f"fcc_{host}_n{n}", host, a0, 2, FCC_BASIS, n)
        add(f"fcc_{host}_imp108", host, a0, 3, FCC_BASIS, 1)

    for host, a0 in BCC_HOSTS.items():
        for n in N_AL_SERIES:
            add(f"bcc_{host}_n{n}", host, a0, 3, BCC_BASIS, n)
        add(f"bcc_{host}_imp128", host, a0, 4, BCC_BASIS, 1)

    generate_potcar_script(base_dir, calcs)
    generate_run_script(base_dir, calcs)
    generate_extract_script(base_dir)

    n_fcc = len(FCC_HOSTS) * (len(N_AL_SERIES) + 1)
    n_bcc = len(BCC_HOSTS) * (len(N_AL_SERIES) + 1)
    print(f"Generated {len(calcs)} calculations in {base_dir}/")
    print(f"  FCC hosts ({len(FCC_HOSTS)}): series 32-atom n=0..4 "
          f"+ 108-atom impurity = {n_fcc}")
    print(f"  BCC hosts ({len(BCC_HOSTS)}): series 54-atom n=0..4 "
          f"+ 128-atom impurity = {n_bcc}")
    print()
    print("Next steps:")
    print("  1. cd DILUTE_AL")
    print("  2. bash make_potcar.sh   # needs $VASP_PP_PATH")
    print("  3. bash run_all.sh       # needs $VASPBIN (NP=8 default)")
    print("  4. python extract_results.py")


if __name__ == "__main__":
    main()
