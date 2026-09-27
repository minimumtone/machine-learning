"""Build HEA_ml_numeric_highconf_v2.csv from the original MPEA dataset.

v2 fixes the noise ceiling of v1 (HEA_ml_numeric_highconf.csv):
- adds condition/microstructure features from the source dataset
  (log10 grain size + missing flag, calculated density, per-phase
  microstructure flags such as B2/L1$_2$/Laves, number of phases),
- drops the leak column ys_log10 and the coarse phase_* columns
  superseded by micro_*,
- preserves missingness in physical and microstructure descriptors; fold-aware
  imputation is performed later in ``pipeline.py``,
- aggregates rows whose full feature vectors are identical
  (median yield strength) since they are indistinguishable to any model.

Source: Borg et al., "Expanded dataset of mechanical properties and
observed phases of multi-principal element alloys", Sci. Data 7, 430 (2020).
https://github.com/CitrineInformatics/MPEA_dataset

Usage:
    python build_highconf_v2.py /path/to/MPEA_dataset.csv
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ELEMENTS = ["Al", "C", "Co", "Cr", "Cu", "Fe", "Hf", "Mn", "Mo", "Nb",
            "Ni", "Si", "Ta", "Ti", "V", "Zr"]
TEMP_COL = "PROPERTY: Test temperature ($^\\circ$C)"
GRAIN_COL = "PROPERTY: grain size ($\\mu$m)"
DENS_COL = "PROPERTY: Calculated Density (g/cm$^3$)"
PROC_MAP = {"ANNEAL": "ANNEAL", "CAST": "CAST", "WROUGHT": "WROUGHT",
            "OTHER": "OTHER", "POWDER": "OTHER"}


def derive_microstructure(micro: pd.Series) -> pd.DataFrame:
    """Derive phase flags while preserving missing microstructure values."""
    text = micro.astype("string")
    missing = text.isna() | text.str.strip().str.lower().eq("unknown")
    out = pd.DataFrame(index=micro.index)
    out["micro_missing"] = missing.astype(float)
    for phase in ["FCC", "BCC", "B2", "L12", "HCP", "Sec.", "Laves"]:
        col = "micro_" + phase.replace(".", "")
        out[col] = text.str.contains(
            re.escape(phase), na=False,
        ).astype(float).mask(missing)
    out["micro_n_phases"] = (
        text.str.count(r"\+").add(1.0).where(~missing)
    )
    return out


def parse_formula(formula: str) -> dict:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9.]*)", str(formula))
    counts = {el: float(n) if n else 1.0 for el, n in tokens if el}
    total = sum(counts.values())
    return {el: v / total for el, v in counts.items()}


def main(mpea_path: str) -> None:
    old = pd.read_csv(HERE / "HEA_ml_numeric_highconf.csv")
    src = pd.read_csv(mpea_path)
    src.columns = [c.strip() for c in src.columns]

    sub = src[(src["PROPERTY: Type of test"] == "T")
              & (src[TEMP_COL].between(15, 30))]
    sub = sub[sub["PROPERTY: YS (MPa)"].notna()].reset_index(drop=True)
    if len(sub) != len(old):
        raise SystemExit(
            f"row mismatch: source subset {len(sub)} vs v1 {len(old)}")

    fractions = pd.DataFrame(
        [{f"{e}_frac": parse_formula(f).get(e, 0.0) for e in ELEMENTS}
         for f in sub["FORMULA"]])

    def key(fracs: np.ndarray, ys: float, year: float, proc: str,
            temp: float) -> tuple:
        return (tuple(np.round(fracs, 4)) + (round(float(ys), 1),
                int(year), str(proc), round(float(temp), 1)))

    frac_cols = [f"{e}_frac" for e in ELEMENTS]
    proc_flags = ["ANNEAL", "CAST", "OTHER", "WROUGHT"]
    pool = defaultdict(list)
    for j in range(len(sub)):
        pool[key(fractions.loc[j].values,
                 sub.loc[j, "PROPERTY: YS (MPa)"],
                 sub.loc[j, "REFERENCE: year"],
                 PROC_MAP.get(sub.loc[j, "PROPERTY: Processing method"],
                              "OTHER"),
                 sub.loc[j, TEMP_COL])].append(j)
    match = []
    for i in range(len(old)):
        proc = next((p for p in proc_flags
                     if old.loc[i, f"processing_{p}"] == 1.0), "OTHER")
        k = key(old.loc[i, frac_cols].values,
                old.loc[i, "yield_strength_MPa"],
                old.loc[i, "year"], proc,
                old.loc[i, "temperature_C"])
        if not pool[k]:
            raise SystemExit(f"v1 row {i} not found in source")
        match.append(pool[k].pop(0))
    sub = sub.iloc[match].reset_index(drop=True)

    grain = pd.to_numeric(sub[GRAIN_COL], errors="coerce")
    micro = sub["PROPERTY: Microstructure"]
    dens = pd.to_numeric(sub[DENS_COL], errors="coerce")

    new = old.copy()
    new["grain_size_um_log10"] = np.log10(grain)
    new["grain_size_missing"] = grain.isna().astype(float)
    new["calc_density_gcm3"] = dens
    new = pd.concat([new, derive_microstructure(micro)], axis=1)
    new = new.drop(columns=["ys_log10", "phase_BCC", "phase_FCC",
                            "phase_other"])

    feat_cols = [c for c in new.columns if c != "yield_strength_MPa"]
    agg = new.groupby(feat_cols, as_index=False,
                      dropna=False)["yield_strength_MPa"].median()
    out = HERE / "HEA_ml_numeric_highconf_v2.csv"
    agg.to_csv(out, index=False)
    print(f"wrote {out} ({len(agg)} rows, {len(agg.columns)} cols)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
