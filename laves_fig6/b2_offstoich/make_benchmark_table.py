#!/usr/bin/env python3
"""Benchmark MACE-MP-0 medium against MP PBE and experimental lattice constants.

Outputs `analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`.

For cubic phases the table shows the conventional lattice constant a; for
non-cubic phases the three independent cell edges (sorted ascending) are shown
so that the cell shape is captured without assuming a particular axis
labelling convention.
"""
import json
import os
import numpy as np
import pandas as pd
from ase.io import read
import spglib

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

mace = pd.read_csv(os.path.join(AN, "mace_mp_ref_results.csv"))
with open(os.path.join(AN, "mp_reference_structures.json")) as f:
    mp = {r["label"]: r for r in json.load(f)}

# experimental conventional lattice constants (sorted ascending a <= b <= c).
# Cubic phases have a == b == c.
exp = {
    "Ni": {"lattice": (3.524, 3.524, 3.524), "source": "exp fcc-Ni"},
    "Al": {"lattice": (4.050, 4.050, 4.050), "source": "exp fcc-Al"},
    "B2_NiAl": {"lattice": (2.887, 2.887, 2.887), "source": "Yamanouchi Fig. 6(a)"},
    "L12_Ni3Al": {"lattice": (3.572, 3.572, 3.572), "source": "exp L1$_2$-Ni$_3$Al"},
    # Khadkikar & Vedula (binary Ni65.3Al34.7), Pt5Ga3-type orthorhombic:
    # conventional cell a=7.475 Å, b=3.732 Å, c=6.727 Å, sorted ascending below.
    "Ni5Al3": {"lattice": (3.732, 6.727, 7.475), "source": "Khadkikar & Vedula; Pt5Ga3-type"},
    "Ni2Al3": {"lattice": (4.036, 4.036, 4.888), "source": "Bradley & Taylor 1937; hP5 P$\\bar{3}$m1"},
    "NiAl3": {"lattice": (4.811, 6.613, 7.367), "source": "Viklund, Hau\u00dfmann & Lidin 1996; Pnma"},
    "Ni3Al4": {"lattice": (11.408, 11.408, 11.408), "source": "Bradley & Taylor 1937; I$\\bar{4}$3d"},
}


def conventional_sorted_lengths(atoms):
    """Return the three conventional cell edges in ascending order.

    For primitive/high-symmetry cells returned by ASE and Materials Project,
    spglib is used to obtain the standard conventional cell.  Where spglib
    fails, the original ASE primitive handling (fcc/bcc-rhombohedral) is used.
    """
    lengths = atoms.get_cell().lengths()
    angles = atoms.get_cell().angles()
    n = len(atoms)

    # Try spglib conventional standardisation first.
    try:
        lattice = atoms.get_cell().array
        positions = atoms.get_scaled_positions()
        numbers = atoms.get_atomic_numbers()
        std = spglib.standardize_cell((lattice, positions, numbers),
                                       to_primitive=False, no_idealize=False)
        if std is None:
            std = spglib.standardize_cell((lattice, positions, numbers),
                                           to_primitive=False, no_idealize=True)
        if std is not None:
            conv_lengths = np.sqrt(np.sum(std[0]**2, axis=1))
            return tuple(sorted(conv_lengths))
    except Exception:
        pass

    # fcc 1-atom primitive (angles ~ 60 degrees)
    if n == 1 and all(abs(a - 60.0) < 5.0 for a in angles):
        a_conv = lengths[0] * np.sqrt(2.0)
        return (a_conv, a_conv, a_conv)
    # bcc-like rhombohedral primitive (angles ~ 109.47 degrees)
    if (all(abs(li - lengths[0]) < 1e-3 for li in lengths) and
        all(abs(ai - 109.47) < 2.0 for ai in angles)):
        a_conv = 2.0 * lengths[0] / np.sqrt(3.0)
        return (a_conv, a_conv, a_conv)
    # orthorhombic / tetragonal / trigonal conventional cells
    return tuple(sorted(lengths))


def read_lengths(path):
    if not path or not os.path.exists(path):
        return None
    try:
        atoms = read(path)
        return conventional_sorted_lengths(atoms)
    except Exception:
        return None


def fmt_tuple(t, ndigits=3):
    if t is None:
        return ("—", "—", "—")
    return tuple(str(round(float(v), ndigits)) for v in t)


rows = []
for _, r in mace.iterrows():
    name = r.label
    m = mp.get(name, {})
    mace_tuple = read_lengths(os.path.join(AN, f"mace_{name}.extxyz"))
    mp_tuple = read_lengths(m.get("structure_file"))
    exp_tuple = exp.get(name, {}).get("lattice")

    if exp_tuple and mace_tuple:
        # signed percent error for each axis, then mean absolute
        err = [100.0 * (mm - ee) / ee for mm, ee in zip(mace_tuple, exp_tuple)]
        err_str = " / ".join(f"{e:+.2f}" for e in err)
        mean_err = round(np.mean(np.abs(err)), 2)
    else:
        err_str = "—"
        mean_err = np.nan

    structure_disp = name
    for s, repl in [("B2_NiAl", "B2-NiAl"), ("L12_Ni3Al", "L1$_2$-Ni$_3$Al"),
                    ("Ni3Al4", "Ni$_3$Al$_4$"), ("Ni5Al3", "Ni$_5$Al$_3$"),
                    ("Ni2Al3", "Ni$_2$Al$_3$"), ("NiAl3", "NiAl$_3$")]:
        structure_disp = structure_disp.replace(s, repl)

    m_a, m_b, m_c = fmt_tuple(mace_tuple)
    p_a, p_b, p_c = fmt_tuple(mp_tuple)
    e_a, e_b, e_c = fmt_tuple(exp_tuple)

    rows.append({
        "Structure": structure_disp,
        "x_Al": r.n_Al / r.n_atoms,
        "MACE a (Å)": m_a,
        "MACE b (Å)": m_b,
        "MACE c (Å)": m_c,
        "MP PBE a (Å)": p_a,
        "MP PBE b (Å)": p_b,
        "MP PBE c (Å)": p_c,
        "Exp a (Å)": e_a,
        "Exp b (Å)": e_b,
        "Exp c (Å)": e_c,
        "MACE err vs exp (%)": err_str,
        "|err| mean (%)": round(mean_err, 2) if not np.isnan(mean_err) else "—",
        "MACE E_f (eV/atom)": round(r.formation_energy_per_atom_eV, 3),
        "MP E_f (eV/atom)": round(m.get("formation_energy_per_atom_eV", np.nan), 3) if m else "—",
    })

df = pd.DataFrame(rows)

cols = ["Structure", "x_Al",
        "MACE a (Å)", "MACE b (Å)", "MACE c (Å)",
        "MP PBE a (Å)", "MP PBE b (Å)", "MP PBE c (Å)",
        "Exp a (Å)", "Exp b (Å)", "Exp c (Å)",
        "MACE err vs exp (%)", "|err| mean (%)",
        "MACE E_f (eV/atom)", "MP E_f (eV/atom)"]
df = df[cols]


def to_md(d):
    cols = list(d.columns)
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    lines = [header, sep]
    for _, r in d.iterrows():
        lines.append(" | ".join(str(v) for v in r))
    return "\n".join(lines)


md = to_md(df)
out_path = os.path.join(AN, "BENCHMARK_MACE_vs_MP_vs_EXP.md")
with open(out_path, "w") as f:
    f.write("# MACE-MP-0 medium benchmark: Ni/Al/binary Ni-Al compounds\n\n")
    f.write("References: MP PBE = Materials Project PBE-GGA entries; "
            "Exp = accepted experimental lattice constants at room temperature.\n\n")
    f.write("Non-cubic phases show the three cell edges sorted ascending; "
            "axis labels (a,b,c) may not match a particular crystallographic setting.\n\n")
    f.write(md)
    f.write("\n")
print(f"Wrote {out_path}")
print(df.to_string(index=False))
