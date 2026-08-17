#!/usr/bin/env python3
"""Generate a Yamanouchi & Miura Table 4 analogue using MACE-relaxed B2/fcc data.

References:
- fcc-C12 diameter: D_C12 = (sqrt(2) * V_atom)^(1/3)
- CN8 correction:  D_CN8 = D_C12 / 1.03  (Yamanouchi's Edwards-style conversion)
- B2 nearest-neighbor distance: D_B2 = sqrt(3)/2 * a_B2
"""
import os, pandas as pd, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

b2 = pd.read_csv(os.path.join(AN, "mace_table4_results.csv"))
fcc = pd.read_csv(os.path.join(AN, "fcc_xal_sqs_volumes.csv"))

rows = []
for _, r in b2.iterrows():
    if not r.label.startswith("B2_"):
        continue
    elem = r.formula.replace("Al", "")
    conv_a = (r.volume_A3 / (r.n_atoms / 2.0)) ** (1.0 / 3.0)
    d_b2 = np.sqrt(3.0) / 2.0 * conv_a
    rows.append({
        "System": r.formula,
        "B2 a (Å)": round(conv_a, 4),
        "B2 V/atom (Å³)": round(r.volume_per_atom_A3, 3),
        "D_B2 (CN8, Å)": round(d_b2, 4),
    })

# fcc X(Al) at x=0.5
fcc_rows = {}
for _, r in fcc.iterrows():
    elem = r.X
    v_at = r.volume_per_atom_A3
    d_c12 = (np.sqrt(2.0) * v_at) ** (1.0 / 3.0)
    d_cn8 = d_c12 / 1.03
    fcc_rows[elem] = {
        "fcc V/atom (Å³)": round(v_at, 3),
        "D_fcc (CN12, Å)": round(d_c12, 4),
        "D_fcc (CN8, Å)": round(d_cn8, 4),
    }

# merge
merged = []
for r in rows:
    elem = r["System"].replace("Al", "")
    f = fcc_rows.get(elem, {})
    r2 = {**r, **f}
    r2["ΔD = D_fcc(CN8) - D_B2 (Å)"] = round(f.get("D_fcc (CN8, Å)", np.nan) - r["D_B2 (CN8, Å)"], 4) if "D_fcc (CN8, Å)" in f else "—"
    merged.append(r2)

df = pd.DataFrame(merged)
out_md = os.path.join(AN, "TABLE4_MACE_ANALOGUE.md")
with open(out_md, "w") as f:
    f.write("# Yamanouchi & Miura Table 4 analogue (MACE-MP-0 medium)\n\n")
    f.write("B2/fcc-derived atomic diameters (CN8) for XAl aluminides.\n\n")
    cols = list(df.columns)
    f.write(" | ".join(cols) + "\n")
    f.write(" | ".join(["---"] * len(cols)) + "\n")
    for _, r in df.iterrows():
        f.write(" | ".join(str(v) for v in r) + "\n")
print(f"Wrote {out_md}")
print(df.to_string(index=False))
