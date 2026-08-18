#!/usr/bin/env python3
"""Direct extraction of B2-NiAl first-nearest-neighbor pair interactions using icet.

Uses MACE-MP-0 relaxed bcc-based fully-occupied structures (B2 order-parameter
sweep, antisite branches) and maps them onto a common B2 primitive cell.
Cluster expansions with 1NN, 1NN+2NN, and 1NN+2NN+triplet clusters are fit by
ordinary least squares, and the effective first-nearest-neighbor pair ordering
strength V is extracted from the B2 vs. random-A2 energy difference.

Outputs:
  analysis/icet_b2_cluster_expansion_summary.json
  analysis/icet_b2_predictions.csv
  figures/fig_icet_ce_parity.png
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.tools import map_structure_to_reference
import icet.input_output.logging_tools as lt

lt.set_log_config(level="ERROR")

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
NIALL_EXT = os.path.join(BASE, "..", "niall_ext")
for d in (AN, FIG):
    os.makedirs(d, exist_ok=True)

# Reference B2 primitive (conventional 2-atom cell) at the MACE perfect B2 volume.
A_REF = 2.882456714930723
PRIM = Atoms(
    symbols="NiAl",
    scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    cell=[A_REF, A_REF, A_REF],
    pbc=True,
)
SUBLATTICES = [["Ni", "Al"], ["Ni", "Al"]]
EXPECTED_N = 128


def load_structure_inputs():
    """Return list of (atoms, energy_per_atom, tag, source) for fully occupied bcc configs."""
    inputs = []

    # B2 order-parameter sweep (eta=1.0 -> perfect B2, eta=0.0 -> A2-like at x=0.5)
    order_csv = os.path.join(NIALL_EXT, "analysis", "b2_order_param.csv")
    order = pd.read_csv(order_csv)
    order = order[order.converged == True]
    for _, r in order.iterrows():
        fn = os.path.join(NIALL_EXT, "relax", f"{r.structure_id}.extxyz")
        if not os.path.exists(fn):
            continue
        at = read(fn)
        at.wrap()
        inputs.append((at, r.energy_eV / r.n_atoms, r.structure_id, "b2_order_param"))

    # Antisite / perfect branches from the off-stoichiometry runs.
    for csv in glob.glob(os.path.join(BASE, "analysis", "b2_offstoich_volumes*.csv")):
        df = pd.read_csv(csv)
        df = df[df.converged == True]
        for _, r in df.iterrows():
            if r.branch not in ("antisite", "perfect"):
                continue
            fn = os.path.join(BASE, "relax", f"{r.structure_id}.extxyz")
            if not os.path.exists(fn):
                continue
            at = read(fn)
            at.wrap()
            inputs.append(
                (at, r.energy_eV / r.n_atoms, r.structure_id, os.path.basename(csv))
            )

    return inputs


def map_and_filter_inputs(inputs):
    """Map all candidate structures to the common reference primitive.

    Returns (mapped_inputs, skipped_counts) where mapped_inputs is a list of
    (atoms, energy_per_atom, tag, source) for accepted structures.
    """
    mapped_inputs = []
    skipped = {"vacancy_X": 0, "wrong_n_sites": 0, "mapping_error": 0}
    for at, e_per_atom, tag, source in inputs:
        try:
            mapped, _ = map_structure_to_reference(
                at,
                PRIM,
                tol_positions=0.5,
                suppress_warnings=True,
                assume_no_cell_relaxation=False,
            )
        except Exception as exc:
            skipped["mapping_error"] += 1
            print(f"[skip {tag}] map_structure_to_reference failed: {exc}")
            continue
        if "X" in mapped.get_chemical_symbols():
            skipped["vacancy_X"] += 1
            print(f"[skip {tag}] mapped structure contains X vacancies")
            continue
        if len(mapped) != EXPECTED_N:
            skipped["wrong_n_sites"] += 1
            print(f"[skip {tag}] mapped n_sites={len(mapped)} != {EXPECTED_N}")
            continue
        mapped_inputs.append((mapped, e_per_atom, tag, source))
    return mapped_inputs, skipped


def add_to_structure_container(sc, mapped_inputs):
    """Add pre-mapped structures to StructureContainer and return metadata."""
    accepted = []
    for mapped, e_per_atom, tag, source in mapped_inputs:
        sc.add_structure(mapped, properties={"energy": e_per_atom}, user_tag=tag)
        accepted.append((tag, source, e_per_atom))
    return accepted


def loocv_rmse(A, y):
    """Leave-one-out cross-validation RMSE using the pseudo-inverse."""
    n, p = A.shape
    if n <= p:
        return np.nan
    ATA_inv = np.linalg.pinv(A.T @ A)
    H = A @ ATA_inv @ A.T
    # Leverage; guard against H_ii == 1
    leverage = np.clip(np.diag(H), None, 1.0 - 1e-12)
    residuals = y - A @ (ATA_inv @ (A.T @ y))
    press = np.mean((residuals / (1.0 - leverage)) ** 2)
    return np.sqrt(press)


def random_a2_energy(ce, n_seeds=120):
    """Average CE energy of random A2 configurations at x_Al=0.5 on the 4x4x4 supercell."""
    rng = np.random.default_rng(42)
    energies = []
    for _ in range(n_seeds):
        rand = PRIM.repeat((4, 4, 4))
        syms = np.array(rand.get_chemical_symbols())
        rng.shuffle(syms)
        rand.set_chemical_symbols(syms)
        energies.append(ce.predict(rand))
    return float(np.mean(energies)), float(np.std(energies))


def fit_ce(label, cutoffs, inputs):
    """Fit a cluster expansion and return a dict of results."""
    cs = ClusterSpace(PRIM, cutoffs, SUBLATTICES)
    sc = StructureContainer(cs)
    accepted = add_to_structure_container(sc, inputs)

    A, y = sc.get_fit_data(key="energy")
    params, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ params
    residual = y - pred
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_abs_res = float(np.max(np.abs(residual)))
    loo_rmse = float(loocv_rmse(A, y))

    ce = ClusterExpansion(cs, params)
    e_b2_per_atom = float(ce.predict(PRIM))
    e_a2_rand_per_atom, e_a2_rand_std = random_a2_energy(ce, n_seeds=120)

    # Effective ordering strength V = -DeltaE_order / 4, with DeltaE per formula.
    delta_order_per_atom = e_a2_rand_per_atom - e_b2_per_atom
    V_eff = -delta_order_per_atom / 2.0

    # Per-orbit information from icet.
    orbit_df = cs.to_dataframe()
    orbit_info = []
    for i, row in orbit_df.iterrows():
        idx = int(row["orbit_index"])
        # Note: p vector starts at orbit 0 after the zero-body term.
        pi = float(params[idx + 1]) if idx + 1 < len(params) else 0.0
        orbit_info.append(
            {
                "order": int(row["order"]),
                "radius": float(row["radius"]),
                "multiplicity": int(row["multiplicity"]),
                "eci_eV": pi,
            }
        )

    # For the 1NN-only model, also extract Ising J values from endpoint predictions.
    J = None
    V_pair = None
    d_1nn = A_REF * np.sqrt(3.0) / 2.0
    d_2nn = A_REF
    if len(cutoffs) == 1 and cutoffs[0] < (d_2nn - 0.1):
        b2 = PRIM.copy()
        ni = PRIM.copy()
        ni.set_chemical_symbols(["Ni", "Ni"])
        al = PRIM.copy()
        al.set_chemical_symbols(["Al", "Al"])

        e_b2 = 2.0 * ce.predict(b2)
        e_ni = 2.0 * ce.predict(ni)
        e_al = 2.0 * ce.predict(al)
        J_NiAl = e_b2 / 8.0
        J_NiNi = e_ni / 8.0
        J_AlAl = e_al / 8.0
        V_pair = J_NiAl - (J_NiNi + J_AlAl) / 2.0
        J = {
            "J_NiAl_eV": round(J_NiAl, 6),
            "J_NiNi_eV": round(J_NiNi, 6),
            "J_AlAl_eV": round(J_AlAl, 6),
            "V_pair_eV": round(V_pair, 6),
        }

    predictions = []
    for (tag, source, e_true), e_pred in zip(accepted, pred):
        predictions.append(
            {
                "structure_id": tag,
                "source": source,
                "energy_per_atom_true_eV": e_true,
                "energy_per_atom_pred_eV": float(e_pred),
                "residual_eV": float(e_true - e_pred),
            }
        )

    return {
        "label": label,
        "cutoffs_A": cutoffs,
        "n_structures": len(accepted),
        "n_parameters": int(len(params)),
        "rmse_eV_per_atom": rmse,
        "max_abs_residual_eV_per_atom": max_abs_res,
        "loocv_rmse_eV_per_atom": loo_rmse,
        "e_B2_per_atom_eV": e_b2_per_atom,
        "e_A2_random_per_atom_eV": e_a2_rand_per_atom,
        "e_A2_random_std_eV": e_a2_rand_std,
        "V_eff_eV_per_bond": V_eff,
        "V_pair_eV_per_bond": V_pair,
        "J_values_eV": J,
        "parameters": params.tolist(),
        "orbits": orbit_info,
        "predictions": predictions,
    }


def plot_parity(results, out_png):
    """Multi-panel parity plot for each CE model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        true = np.array([p["energy_per_atom_true_eV"] for p in res["predictions"]])
        pred = np.array([p["energy_per_atom_pred_eV"] for p in res["predictions"]])
        ax.scatter(true, pred, s=30, alpha=0.6, edgecolors="k", linewidths=0.3)
        lo = min(true.min(), pred.min()) - 0.05
        hi = max(true.max(), pred.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("MACE energy / atom (eV)", fontsize=13)
        ax.set_ylabel("CE prediction / atom (eV)", fontsize=13)
        ax.set_title(
            f"{res['label']}: RMSE={res['rmse_eV_per_atom']:.4f} eV\n"
            f"V={res['V_eff_eV_per_bond']:.4f} eV/bond",
            fontsize=13,
        )
        ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[parity] saved {out_png}")


def main():
    inputs = load_structure_inputs()
    print(f"[load] {len(inputs)} candidate structures loaded")
    mapped_inputs, skipped = map_and_filter_inputs(inputs)
    print(f"[map] accepted={len(mapped_inputs)} skipped={skipped}")

    d_1nn = A_REF * np.sqrt(3.0) / 2.0  # ~2.496 A
    d_2nn = A_REF  # ~2.882 A
    models = [
        ("1NN_pairs", [d_1nn + 0.07]),     # only 1NN pair orbit (n_par=3)
        ("1NN+2NN_pairs", [d_2nn + 0.07]), # add 2NN pair orbit (n_par=4)
        ("1NN+2NN+triplets", [d_2nn + 0.07, d_2nn + 0.07]),  # add triplet orbit (n_par=5)
    ]

    results = []
    for label, cutoffs in models:
        res = fit_ce(label, cutoffs, mapped_inputs)
        results.append(res)
        print(
            f"[{label}] N={res['n_structures']} npar={res['n_parameters']} "
            f"RMSE={res['rmse_eV_per_atom']:.4f} "
            f"max|res|={res['max_abs_residual_eV_per_atom']:.4f} "
            f"V={res['V_eff_eV_per_bond']:.4f} eV/bond"
        )
        if res["V_pair_eV_per_bond"] is not None:
            print(f"        V_pair(1NN)={res['V_pair_eV_per_bond']:.4f} eV/bond")

    # Save summary JSON (predictions are stored separately to keep JSON small).
    summary = {
        "reference_a_A": A_REF,
        "skipped_after_mapping": skipped,
        "models": [
            {
                k: v
                for k, v in r.items()
                if k != "predictions" and k != "parameters"
            }
            for r in results
        ],
    }
    summary["comparison"] = {
        "V_from_ordering_eV_per_bond": round(-0.3509 / 4.0, 6),
        "V_pair_constant_eV_per_bond": -0.1449,
        "icet_1NN_V_pair_eV_per_bond": round(results[0]["V_pair_eV_per_bond"], 6),
        "icet_2NN_triplets_V_eff_eV_per_bond": round(results[2]["V_eff_eV_per_bond"], 6),
        "interpretation": (
            "1NN-only pair model gives V ~ -0.14 eV/bond, close to the "
            "constant-pair estimate from isolated point defects. Adding 2NN "
            "same-sublattice pairs and triplets reduces the effective ordering "
            "strength to ~ -0.10 eV/bond, approaching the thermodynamic "
            "V = -0.088 eV/bond from the B2/A2 energy difference."
        ),
    }
    summary_path = os.path.join(AN, "icet_b2_cluster_expansion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] saved {summary_path}")

    # Save predictions for all models in one CSV.
    rows = []
    for res in results:
        for p in res["predictions"]:
            rows.append(
                {
                    "model": res["label"],
                    "structure_id": p["structure_id"],
                    "source": p["source"],
                    "energy_per_atom_true_eV": p["energy_per_atom_true_eV"],
                    "energy_per_atom_pred_eV": p["energy_per_atom_pred_eV"],
                    "residual_eV": p["residual_eV"],
                }
            )
    pred_csv = os.path.join(AN, "icet_b2_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    print(f"[predictions] saved {pred_csv}")

    parity_png = os.path.join(FIG, "fig_icet_ce_parity.png")
    plot_parity(results, parity_png)


if __name__ == "__main__":
    main()
