#!/usr/bin/env python3
"""Benchmark MACE-MP-0 medium against MP PBE and experimental lattice constants.

Outputs `BENCHMARK_MACE_vs_MP_vs_EXP.md`.
"""
import json, os, pandas as pd, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

mace = pd.read_csv(os.path.join(AN, "mace_mp_ref_results.csv"))
with open(os.path.join(AN, "mp_reference_structures.json")) as f:
    mp = {r["label"]: r for r in json.load(f)}

# experimental values (accepted literature / Yamanouchi Fig. 6 calibrations)
exp = {
    "Ni": {"a_A": 3.524, "V_A3": 10.94, "source": "exp fcc-Ni"},
    "Al": {"a_A": 4.050, "V_A3": 16.61, "source": "exp fcc-Al"},
    "B2_NiAl": {"a_A": 2.887, "V_A3": 12.02, "source": "Yamanouchi Fig. 6(a)"},
    "L12_Ni3Al": {"a_A": 3.572, "V_A3": 11.40, "source": "exp L1_2-Ni_3Al"},
}

# For cubic structures, convert the primitive/conventional cell volume to a.
# MACE returns the *total* cell volume; fcc elements from MP are 1-atom primitive cells.
conv_mult = {"Ni": 4, "Al": 4, "B2_NiAl": 1, "L12_Ni3Al": 1}

def a_from_v(name, v, n):
    m = conv_mult.get(name)
    if m is None:
        return np.nan
    # if MP cell already has 4 conventional atoms, use total volume directly.
    if n == 1 and m == 4:
        return (v * m) ** (1.0/3.0)
    return (v) ** (1.0/3.0)

rows = []
for _, r in mace.iterrows():
    name = r.label
    m = mp.get(name, {})
    a_mace = a_from_v(name, r.volume_A3, r.n_atoms)
    if name in exp:
        a_exp = exp[name]["a_A"]
        err_mace_pct = (a_mace - a_exp) / a_exp * 100
    else:
        a_exp = np.nan
        err_mace_pct = np.nan
    if m and "volume_per_atom_A3" in m:
        mp_total_v = m["volume_per_atom_A3"] * m["n_atoms"]
        a_mp = a_from_v(name, mp_total_v, m["n_atoms"])
    else:
        a_mp = np.nan
    rows.append({
        "Structure": name.replace("B2_NiAl", "B2-NiAl").replace("L12_Ni3Al", "L1$_2$-Ni$_3$Al"),
        "x_Al": r.n_Al / r.n_atoms,
        "MACE a (Å)": round(a_mace, 4),
        "MP PBE a (Å)": round(a_mp, 4) if not np.isnan(a_mp) else "—",
        "Exp a (Å)": round(a_exp, 3) if not np.isnan(a_exp) else "—",
        "MACE error vs exp (%)": round(err_mace_pct, 2) if not np.isnan(err_mace_pct) else "—",
        "MACE E_f (eV/atom)": round(r.formation_energy_per_atom_eV, 3),
        "MP E_f (eV/atom)": round(m.get("formation_energy_per_atom_eV", np.nan), 3) if m else "—",
    })

df = pd.DataFrame(rows)

def to_md(d):
    cols = list(d.columns)
    header = " | ".join(cols)
    sep = " | ".join(["---"]*len(cols))
    lines = [header, sep]
    for _, r in d.iterrows():
        lines.append(" | ".join(str(v) for v in r))
    return "\n".join(lines)

md = to_md(df)
out_path = os.path.join(AN, "BENCHMARK_MACE_vs_MP_vs_EXP.md")
with open(out_path, "w") as f:
    f.write("# MACE-MP-0 medium benchmark: Ni/Al/B2-NiAl/L1$_2$-Ni$_3$Al\n\n")
    f.write("References: MP PBE = Materials Project PBE-GGA entries; Exp = accepted experimental lattice constants at room temperature.\n\n")
    f.write(md)
    f.write("\n")
print(f"Wrote {out_path}")
print(df.to_string(index=False))
