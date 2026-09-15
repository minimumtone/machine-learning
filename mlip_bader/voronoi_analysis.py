#!/usr/bin/env python3
"""Compute periodic Voronoi volumes and binary-to-HEA excess-volume survival."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from scipy.spatial import ConvexHull, Voronoi


ROOT = Path(__file__).resolve().parent
STRUCTURES = ROOT / "structures"
SUMMARY_CSV = ROOT / "relax_results.csv"
PER_ATOM_CSV = ROOT / "voronoi_per_atom.csv"
SUMMARY_MD = ROOT / "voronoi_summary.md"
FIGURE = ROOT / "fig_delta_v_per_element.png"
LOG = ROOT / "voronoi_analysis_output.txt"
TOL = 1e-3


def log_line(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def periodic_voronoi_volumes(atoms) -> np.ndarray:
    """Return central-cell atom volumes from a 3x3x3 periodic image set."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    image_positions = np.concatenate([positions + shift @ cell for shift in shifts])
    central_offset = list(map(tuple, shifts)).index((0.0, 0.0, 0.0)) * len(atoms)
    voronoi = Voronoi(image_positions)
    volumes = np.empty(len(atoms), dtype=float)
    for index in range(len(atoms)):
        region = voronoi.regions[voronoi.point_region[central_offset + index]]
        if not region or -1 in region:
            raise RuntimeError(f"unbounded Voronoi region for atom {index}")
        vertices = voronoi.vertices[region]
        volumes[index] = ConvexHull(vertices).volume
    ratio = volumes.sum() / atoms.get_volume()
    if abs(ratio - 1.0) > TOL:
        raise RuntimeError(
            f"Voronoi volume sum mismatch: sum={volumes.sum():.8f}, "
            f"cell={atoms.get_volume():.8f}, ratio={ratio:.8f}"
        )
    return volumes


def load_atom_volumes() -> pd.DataFrame:
    rows = []
    files = sorted(STRUCTURES.glob("*.extxyz"))
    if not files:
        raise FileNotFoundError(f"No relaxed structures found under {STRUCTURES}")
    for path in files:
        atoms = read(path)
        volumes = periodic_voronoi_volumes(atoms)
        ratio = volumes.sum() / atoms.get_volume()
        log_line(f"{path.name}: Voronoi sum/cell volume = {ratio:.8f}")
        label_seed = path.stem.rsplit("_s", 1)
        label, seed = label_seed[0], int(label_seed[1])
        for atom_index, (element, volume) in enumerate(zip(atoms.get_chemical_symbols(), volumes)):
            rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "atom_index": atom_index,
                    "element": element,
                    "V_vor_A3": volume,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(PER_ATOM_CSV, index=False)
    return frame


def composition(label: str) -> dict[str, float]:
    if label == "HfNbTaTiZr":
        return {"Hf": 26 / 128, "Nb": 26 / 128, "Ta": 26 / 128, "Ti": 25 / 128, "Zr": 25 / 128}
    if label == "AlNbTiV":
        return {element: 0.25 for element in ("Al", "Nb", "Ti", "V")}
    if "-" in label:
        first, second = label.split("-")
        return {first: 0.5, second: 0.5}
    return {label: 1.0}


def make_tables(frame: pd.DataFrame):
    grouped = frame.groupby(["label", "seed", "element"], as_index=False)["V_vor_A3"].mean()
    pure = grouped[grouped.label.isin(["Hf", "Nb", "Ta", "Ti", "Zr", "Al", "V"])]
    pure_values = pure.groupby("element")["V_vor_A3"].mean().to_dict()
    pure_rows = [
        {"element": element, "V_i_pure_A3": pure_values[element]}
        for element in sorted(pure_values)
    ]

    binary_rows = []
    binary_delta: dict[tuple[str, str], float] = {}
    binary_omega: dict[tuple[str, str], float] = {}
    for label in sorted(grouped.label.unique()):
        if "-" not in label:
            continue
        first, second = label.split("-")
        binary = grouped[grouped.label == label]
        means = binary.groupby("element")["V_vor_A3"].mean().to_dict()
        for element, other in ((first, second), (second, first)):
            delta = means[element] - pure_values[element]
            binary_delta[(element, other)] = delta
            binary_rows.append(
                {"pair": label, "element": element, "neighbor": other, "delta_V_A3": delta}
            )
        vveg = (pure_values[first] + pure_values[second]) / 2
        vcell_per_atom = binary["V_vor_A3"].mean()
        omega = vcell_per_atom / vveg - 1
        binary_omega[(first, second)] = omega
        binary_omega[(second, first)] = omega
        binary_rows[-2]["Omega_MACE"] = omega
        binary_rows[-1]["Omega_MACE"] = omega

    hea_rows = []
    cell_rows = []
    for label in ("HfNbTaTiZr", "AlNbTiV"):
        c = composition(label)
        hea = grouped[grouped.label == label]
        v_by_element = hea.groupby("element")["V_vor_A3"]
        vveg = sum(c[element] * pure_values[element] for element in c)
        vcell = sum(c[element] * v_by_element.get_group(element).mean() for element in c)
        denominator = 0.0
        for first, second in itertools.permutations(c, 2):
            denominator += (
                c[first]
                * c[second]
                * (pure_values[first] + pure_values[second])
                * binary_omega[tuple(sorted((first, second)))]
            )
        f_cell = (vcell - vveg) / denominator if abs(denominator) > 1e-12 else np.nan
        cell_rows.append(
            {
                "label": label,
                "V_HEA_per_atom_A3": vcell,
                "V_Vegard_A3": vveg,
                "denominator_A3": denominator,
                "f_cell": f_cell,
            }
        )
        for element in c:
            samples = v_by_element.get_group(element)
            delta = float(samples.mean() - pure_values[element])
            std = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
            predicted = sum(2 * c[neighbor] * binary_delta[(element, neighbor)] for neighbor in c if neighbor != element)
            reliable = abs(predicted) >= 0.05
            hea_rows.append(
                {
                    "label": label,
                    "element": element,
                    "delta_V_HEA_A3": delta,
                    "std_A3": std,
                    "delta_V_pred_A3": predicted,
                    "f_i": delta / predicted if reliable else np.nan,
                    "flag": "" if reliable else "unreliable |pred|<0.05",
                }
            )
    return pure_rows, binary_rows, hea_rows, cell_rows


def write_markdown(pure_rows, binary_rows, hea_rows, cell_rows) -> None:
    lines = [
        "# MACE-MP-0 Voronoi excess-volume analysis",
        "",
        "MACE-MP-0 small, float64, CPU; 128-atom BCC cells; random occupations.",
        "Periodic Voronoi volumes use scipy.spatial.Voronoi on a 3x3x3 image set.",
        "",
        "## 1. Pure-element volumes",
        "",
        "| element | V_i^pure (A^3) |",
        "|---|---:|",
    ]
    lines += [f"| {r['element']} | {r['V_i_pure_A3']:.6f} |" for r in pure_rows]
    lines += ["", "## 2. Binary excess volumes", "", "| pair | element | neighbor | Delta V (A^3) | Omega_MACE |", "|---|---|---|---:|---:|"]
    lines += [
        f"| {r['pair']} | {r['element']} | {r['neighbor']} | {r['delta_V_A3']:.6f} | {r['Omega_MACE']:.6f} |"
        for r in binary_rows
    ]
    for cell in cell_rows:
        lines += [
            "",
            f"## 3. {cell['label']}",
            "",
            "| element | Delta V_HEA (A^3) | seed std (A^3) | Delta V_pred (A^3) | f_i | flag |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in hea_rows:
            if r["label"] == cell["label"]:
                f_value = "—" if np.isnan(r["f_i"]) else f"{r['f_i']:.6f}"
                lines.append(
                    f"| {r['element']} | {r['delta_V_HEA_A3']:.6f} | {r['std_A3']:.6f} | "
                    f"{r['delta_V_pred_A3']:.6f} | {f_value} | {r['flag']} |"
                )
        lines += [
            "",
            f"Cell: V_HEA/N = {cell['V_HEA_per_atom_A3']:.6f} A^3; "
            f"V_Vegard = {cell['V_Vegard_A3']:.6f} A^3; "
            f"Alonso full-additivity denominator = {cell['denominator_A3']:.6f} A^3; "
            f"f_cell = {cell['f_cell']:.6f}.",
        ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(hea_rows) -> None:
    plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for axis, label in zip(axes, ("HfNbTaTiZr", "AlNbTiV")):
        rows = [row for row in hea_rows if row["label"] == label]
        x = np.arange(len(rows))
        width = 0.38
        axis.bar(x - width / 2, [row["delta_V_pred_A3"] for row in rows], width, label="Predicted")
        axis.bar(x + width / 2, [row["delta_V_HEA_A3"] for row in rows], width, label="HEA Voronoi")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [row["element"] for row in rows])
        axis.set_title(label)
        axis.set_ylabel("Excess Voronoi volume (A$^3$)")
        axis.legend()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    LOG.write_text("Periodic Voronoi analysis log\n", encoding="utf-8")
    frame = load_atom_volumes()
    pure_rows, binary_rows, hea_rows, cell_rows = make_tables(frame)
    write_markdown(pure_rows, binary_rows, hea_rows, cell_rows)
    write_figure(hea_rows)
    log_line(f"Wrote {len(frame)} per-atom rows to {PER_ATOM_CSV}.")
    log_line(f"Wrote summary to {SUMMARY_MD} and figure to {FIGURE}.")
    for cell in cell_rows:
        log_line(f"{cell['label']}: f_cell={cell['f_cell']:.6f}")
    for row in hea_rows:
        log_line(
            f"{row['label']} {row['element']}: delta_HEA={row['delta_V_HEA_A3']:.6f}, "
            f"delta_pred={row['delta_V_pred_A3']:.6f}, f_i={row['f_i']}"
        )


if __name__ == "__main__":
    main()
