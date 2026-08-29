#!/usr/bin/env python3
"""Generate supplementary figures from the frozen paper analysis outputs."""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))

from generate_all_figures import (  # noqa: E402
    compute_omega_sf_pairwise,
    load_compounds,
    load_sqs_data,
)
from hea_lattice_xgboost import (  # noqa: E402
    ALONSO_TABLE2,
    INDEPENDENT_TEST,
    compute_eq10_scaled,
)
from detect_unrelaxed_volumes import (  # noqa: E402
    COMPOSITION_PATTERN,
    VASP_ATOMIC_VOLUMES,
    flagged_row,
)


for fp in fm.findSystemFonts():
    if "ipag" in fp.lower() or "ipagothic" in fp.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
else:
    for fp in fm.findSystemFonts():
        if "wqy" in fp.lower():
            plt.rcParams["font.family"] = (
                fm.FontProperties(fname=fp).get_name()
            )
            break

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "figure.dpi": 200,
})

OUTPUTS = {
    "heatmap": PAPER / "fig_sqs_omega_heatmap.png",
    "q_scan": PAPER / "fig_q_scan.png",
    "transfer": PAPER / "fig_transfer_fraction.png",
    "residual": PAPER / "fig_indep_residual_descriptors.png",
    "chen": PAPER / "fig_chen_screening_hist.png",
    "unrelaxed": PAPER / "fig_unrelaxed_diagnostic.png",
}


def save_figure(fig, name: str, rect=None) -> None:
    """Save one newly named figure with the common paper settings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible",
        )
        fig.tight_layout(rect=rect)
    fig.savefig(OUTPUTS[name], dpi=200)
    plt.close(fig)


def omega_matrix(omega: dict[tuple[str, str], float]):
    elements = sorted({element for pair in omega for element in pair})
    values = np.full((len(elements), len(elements)), np.nan)
    index = {element: i for i, element in enumerate(elements)}
    for (element_a, element_b), value in omega.items():
        i, j = index[element_a], index[element_b]
        values[i, j] = value * 100
        values[j, i] = value * 100
    return elements, np.ma.masked_invalid(values)


def make_heatmap(sqs: dict) -> None:
    """Draw BCC and FCC SQS structure-factor heatmaps."""
    matrices = [
        ("(a) BCC SQS 50:50", sqs["omega_dft"]),
        ("(b) FCC SQS 50:50", sqs["fcc_omega_dft"]),
    ]
    all_values = np.concatenate([
        matrix.compressed() for _, omega in matrices
        for matrix in [omega_matrix(omega)[1]]
    ])
    limit = max(abs(all_values).max(), 1e-9)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    image = None
    for ax, (title, omega) in zip(axes, matrices):
        elements, values = omega_matrix(omega)
        image = ax.imshow(values, cmap=cmap, norm=norm, interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("Element")
        ax.set_ylabel("Element")
        ax.set_xticks(range(len(elements)), elements, rotation=90)
        ax.set_yticks(range(len(elements)), elements)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label(r"$\Omega_\mathrm{sf}$ (%)")
    save_figure(fig, "heatmap", rect=[0, 0, 0.87, 1])


def rmse_curve(omega, heas, q_values):
    selected = [hea for hea in heas if hea["struct"] in ("BCC", "FCC")]
    target = np.array([hea["a_exp"] for hea in selected])
    result = []
    for q_value in q_values:
        prediction = np.array([
            compute_eq10_scaled(hea["comp"], hea["struct"], omega, q_value)
            for hea in selected
        ])
        result.append(float(np.sqrt(np.mean((prediction - target) ** 2))))
    return np.asarray(result)


def make_q_scan(sqs: dict, additivity: dict) -> None:
    """Draw calibration and independent-test RMSE scans over q."""
    b2, l12 = compute_omega_sf_pairwise(load_compounds())
    models = [
        ("BCC SQS + DFT Vegard", sqs["omega_dft"], "BCC", "#1f77b4"),
        ("BCC B2 + King", b2, "BCC", "#d62728"),
        ("FCC L1$_2$ + King", l12, "FCC", "#2ca02c"),
    ]
    q_values = np.linspace(0, 3, 121)
    fig, ax = plt.subplots(figsize=(14, 9))
    for label, omega, structure, color in models:
        train = [h for h in ALONSO_TABLE2 if h["struct"] == structure]
        test = [h for h in INDEPENDENT_TEST if h["struct"] == structure]
        ax.plot(
            q_values, rmse_curve(omega, train, q_values),
            "--", color=color, label=f"{label}, calibration",
        )
        ax.plot(
            q_values, rmse_curve(omega, test, q_values),
            "-", color=color, label=f"{label}, test",
        )
    ax.axvline(1, color="black", linestyle=":", linewidth=1.5, label="q = 1")
    q_lines = [
        (
            "BCC SQS $q_\\mathrm{opt}$",
            additivity["BCC_SQS_DFT_vegard"],
            "#1f77b4",
        ),
        (
            "BCC B2 $q_\\mathrm{opt}$",
            additivity["BCC_B2_King"],
            "#d62728",
        ),
        (
            "FCC L1$_2$ $q_\\mathrm{opt}$",
            additivity["FCC_L12_King"],
            "#2ca02c",
        ),
    ]
    for label, model, color in q_lines:
        ax.axvline(
            model["q_opt_calibration"], color=color, linestyle="-.",
            alpha=0.75, label=label,
        )
    ax.axvline(
        additivity["BCC_SQS_DFT_vegard"]["q_exact"],
        color="#1f77b4", linestyle="--", alpha=0.65,
        label=r"BCC $q_\mathrm{exact}$ = 2",
    )
    ax.axvline(
        additivity["FCC_L12_King"]["q_exact"],
        color="#2ca02c", linestyle="--", alpha=0.65,
        label=r"FCC L1$_2$ $q_\mathrm{exact}$ = 2.652",
    )
    ax.set_xlim(0, 3)
    ax.set_xlabel("q (dimensionless)")
    ax.set_ylabel("Independent-test RMSE (Å)")
    ax.set_title("RMSE scan for the structure-factor correction")
    ax.legend(loc="upper left", ncol=2)
    save_figure(fig, "q_scan")


def make_transfer(additivity: dict) -> None:
    """Draw the fitted transfer fraction for each frozen model."""
    labels = [
        ("BCC SQS (DFT Vegard)", "BCC_SQS_DFT_vegard"),
        ("BCC SQS (King)", "BCC_SQS_King"),
        ("BCC B2 (King)", "BCC_B2_King"),
        ("FCC SQS (DFT Vegard)", "FCC_SQS_DFT_vegard"),
        ("FCC L1$_2$ (King)", "FCC_L12_King"),
    ]
    values = [
        additivity[key]["q_opt_calibration"] / additivity[key]["q_exact"]
        for _, key in labels
    ]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(y, values, color="#4c78a8")
    ax.axvline(
        1, color="black", linestyle="--", linewidth=1.5,
        label="f = 1 (exact pairwise additivity)",
    )
    ax.set_yticks(y, [label for label, _ in labels])
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Transfer fraction $f=q_\mathrm{opt}/q_\mathrm{exact}$ "
        "(dimensionless)"
    )
    ax.set_title("Transfer fraction relative to exact pairwise additivity")
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center",
        )
    ax.set_xlim(0, max(values) + 0.35)
    ax.legend(loc="lower right")
    save_figure(fig, "transfer")


def pair_median(components: dict, omega: dict) -> float:
    pairs = [
        tuple(sorted((element_a, element_b)))
        for i, element_a in enumerate(components)
        for element_b in list(components)[i + 1:]
    ]
    values = [abs(omega[pair]) * 100 for pair in pairs if pair in omega]
    return float(np.median(values)) if values else np.nan


def make_residual(sqs: dict) -> None:
    """Draw independent-test residuals against four descriptors."""
    frame = pd.read_csv(PAPER / "results_independent_test.csv")
    omega_by_structure = {
        "BCC": sqs["omega_dft"],
        "FCC": sqs["fcc_omega_dft"],
    }
    records = []
    for row, hea in zip(frame.to_dict("records"), INDEPENDENT_TEST):
        structure = row["struct"]
        records.append({
            "residual": row["a_dft_eq10_ss"] - row["a_exp"],
            "components": len(hea["comp"]),
            "omega_median": pair_median(
                hea["comp"], omega_by_structure[structure]
            ),
            "a_exp": row["a_exp"],
            "structure": structure,
        })
    data = pd.DataFrame(records)
    colors = {"BCC": "#1f77b4", "FCC": "#d62728"}
    markers = {"BCC": "o", "FCC": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    descriptors = [
        ("components", "Number of components (count)"),
        ("omega_median", r"Median $|\Omega_\mathrm{sf}|$ (%)"),
        ("a_exp", "Experimental lattice constant (Å)"),
    ]
    for ax, (column, xlabel), panel in zip(
        axes.flat, descriptors, ["(a)", "(b)", "(c)"]
    ):
        for structure in ("BCC", "FCC"):
            subset = data[data["structure"] == structure]
            ax.scatter(
                subset[column], subset["residual"], color=colors[structure],
                marker=markers[structure], label=structure, s=70, alpha=0.85,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(panel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residual (predicted − experimental) (Å)")
        ax.legend()
    ax = axes[1, 1]
    positions = [1, 2]
    groups = [
        data.loc[data["structure"] == "BCC", "residual"],
        data.loc[data["structure"] == "FCC", "residual"],
    ]
    ax.boxplot(groups, positions=positions, widths=0.45)
    for position, structure in zip(positions, ("BCC", "FCC")):
        jitter = np.linspace(-0.10, 0.10, len(groups[position - 1]))
        ax.scatter(
            position + jitter, groups[position - 1], color=colors[structure],
            marker=markers[structure], s=60, alpha=0.8, label=structure,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("(d)")
    ax.set_xticks(positions, ["BCC", "FCC"])
    ax.set_xlabel("Structure (category)")
    ax.set_ylabel("Residual (predicted − experimental) (Å)")
    ax.legend()
    save_figure(fig, "residual")


def make_chen() -> None:
    """Draw the Chen et al. screening distributions and comparisons."""
    data = pd.read_csv(PAPER / "results_chen2023_screening.csv")
    phases = ["BCC", "FCC"]
    colors = {"BCC": "#1f77b4", "FCC": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    for phase in phases:
        subset = data.loc[data["phase"] == phase]
        axes[0, 0].hist(
            subset["a_pred_A"], bins=35, alpha=0.55, color=colors[phase],
            label=f"{phase} (n={len(subset)})",
        )
        axes[0, 1].hist(
            subset["shift_percent"], bins=35, alpha=0.55, color=colors[phase],
            label=f"{phase} (n={len(subset)})",
        )
        axes[1, 0].scatter(
            subset["delta_r_percent"], subset["shift_percent"],
            s=12, alpha=0.28, color=colors[phase],
            label=f"{phase} (n={len(subset)})",
        )
        axes[1, 1].scatter(
            subset["a_vegard_A"], subset["a_pred_A"],
            s=12, alpha=0.28, color=colors[phase],
            label=f"{phase} (n={len(subset)})",
        )
    axes[0, 0].set_title("(a)")
    axes[0, 0].set_xlabel("Predicted lattice constant (Å)")
    axes[0, 0].set_ylabel("Count (number)")
    axes[0, 0].legend()
    axes[0, 1].set_title("(b)")
    axes[0, 1].set_xlabel("Shift (%)")
    axes[0, 1].set_ylabel("Count (number)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[1, 0].set_title("(c)")
    axes[1, 0].set_xlabel("Atomic-size mismatch Δr (%)")
    axes[1, 0].set_ylabel("Shift (%)")
    axes[1, 0].legend()
    axes[1, 1].set_title("(d)")
    axes[1, 1].set_xlabel("Vegard lattice constant (Å)")
    axes[1, 1].set_ylabel("Predicted lattice constant (Å)")
    low = min(data["a_vegard_A"].min(), data["a_pred_A"].min())
    high = max(data["a_vegard_A"].max(), data["a_pred_A"].max())
    axes[1, 1].plot(
        [low, high], [low, high], color="black", linestyle="--",
        label="y = x",
    )
    axes[1, 1].legend()
    save_figure(fig, "chen")


def expected_volume(row: dict[str, str]) -> float | None:
    match = COMPOSITION_PATTERN.fullmatch(row["dir"])
    if match is None:
        return None
    element_a, count_a, element_b, count_b = match.groups()
    if (
        element_a not in VASP_ATOMIC_VOLUMES
        or element_b not in VASP_ATOMIC_VOLUMES
    ):
        return None
    try:
        natoms = int(row["natoms"])
    except (TypeError, ValueError):
        return None
    return (
        int(count_a) * VASP_ATOMIC_VOLUMES[element_a]
        + int(count_b) * VASP_ATOMIC_VOLUMES[element_b]
    ) / natoms


def make_unrelaxed(sqs: dict) -> None:
    """Show input-volume matches and their effect on BCC SQS pair values."""
    with (ROOT / "data" / "sqs_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plotted = []
    flagged_keys = set()
    for row in rows:
        if row["status"] != "OK" or not row["volume_A3"].strip():
            continue
        expected = expected_volume(row)
        if expected is None:
            continue
        key = (row["dir"], row["structure_root"], int(row["natoms"]))
        is_flagged = flagged_row(row) is not None
        if is_flagged:
            flagged_keys.add(key)
        plotted.append((
            float(row["volume_A3"]) / int(row["natoms"]),
            expected,
            is_flagged,
        ))
    plotted = np.asarray(plotted, dtype=object)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    normal = plotted[~plotted[:, 2].astype(bool)]
    flagged = plotted[plotted[:, 2].astype(bool)]
    axes[0].scatter(
        normal[:, 1].astype(float), normal[:, 0].astype(float),
        s=12, alpha=0.25, color="#4c78a8", label=f"OK (n={len(normal)})",
    )
    axes[0].scatter(
        flagged[:, 1].astype(float), flagged[:, 0].astype(float),
        s=42, color="#d62728", label=f"Flagged (n={len(flagged)})",
    )
    limits = [
        min(
            plotted[:, 0].astype(float).min(),
            plotted[:, 1].astype(float).min(),
        ),
        max(
            plotted[:, 0].astype(float).max(),
            plotted[:, 1].astype(float).max(),
        ),
    ]
    axes[0].plot(limits, limits, "k--", label="y = x")
    axes[0].set_xlabel("Input-table composition-weighted V/atom (Å³/atom)")
    axes[0].set_ylabel("Recorded V/atom (Å³/atom)")
    axes[0].set_title("(a)")
    axes[0].legend()

    omega = sqs["omega_dft"]
    flagged_pairs = set()
    for row in rows:
        if (
            row["structure_root"] != "BCC_SQS"
            or row["status"] != "OK"
        ):
            continue
        try:
            natoms = int(row["natoms"])
        except (TypeError, ValueError):
            continue
        if natoms != 16:
            continue
        match = COMPOSITION_PATTERN.fullmatch(row["dir"])
        if match is None:
            continue
        element_a, count_a, element_b, count_b = match.groups()
        if int(count_a) != 8 or int(count_b) != 8:
            continue
        pair = tuple(sorted((element_a, element_b)))
        key = (row["dir"], row["structure_root"], natoms)
        if key in flagged_keys:
            flagged_pairs.add(pair)
    flagged_values = [
        abs(omega[pair]) * 100 for pair in flagged_pairs if pair in omega
    ]
    normal_values = [
        abs(value) * 100 for pair, value in omega.items()
        if pair not in flagged_pairs
    ]
    combined = np.asarray(flagged_values + normal_values)
    if len(combined):
        bins = np.linspace(combined.min(), combined.max(), 24)
    else:
        bins = 10
    axes[1].hist(
        normal_values, bins=bins, alpha=0.55, color="#4c78a8",
        label=f"Non-flagged (n={len(normal_values)})",
    )
    axes[1].hist(
        flagged_values, bins=bins, alpha=0.75, color="#d62728",
        label=f"Flagged (n={len(flagged_values)})",
    )
    axes[1].set_xlabel(r"$|\Omega_\mathrm{sf}|$ (%)")
    axes[1].set_ylabel("Count (number)")
    axes[1].set_title("(b)")
    axes[1].legend()
    save_figure(fig, "unrelaxed")


def main() -> None:
    sqs = load_sqs_data()
    with (PAPER / "pairwise_additivity_metrics.json").open() as handle:
        additivity = json.load(handle)["models"]
    make_heatmap(sqs)
    make_q_scan(sqs, additivity)
    make_transfer(additivity)
    make_residual(sqs)
    make_chen()
    make_unrelaxed(sqs)
    for output in OUTPUTS.values():
        print(output)


if __name__ == "__main__":
    main()
