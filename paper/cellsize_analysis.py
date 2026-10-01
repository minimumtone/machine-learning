#!/usr/bin/env python3
"""Reproduce the 16- versus 128-atom BCC-SQS cell-size comparison."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(PAPER))

from generate_all_figures import EXCLUDE_ELEMENTS  # noqa: E402
from detect_unrelaxed_volumes import flagged_row  # noqa: E402

DATA = ROOT / "data" / "sqs_results.csv"
METRICS = PAPER / "cellsize_metrics.json"
FIGURE = PAPER / "fig_cellsize_bcc_sqs.png"

# These elements are excluded from the paper's BCC/FCC validation scope
# because their stable structures are incompatible with the enforced BCC/FCC
# reference. Rare-earth and Y exclusions come from the shared analysis source.
STRUCTURE_MISMATCH_ELEMENTS = {"B", "Ge", "Si"}
CELL_SIZE_EXCLUSIONS = set(EXCLUDE_ELEMENTS) | STRUCTURE_MISMATCH_ELEMENTS


def parse_dir(value: str):
    pairs = re.findall(r"([A-Z][a-z]?)(\d+)", str(value))
    if len(pairs) != 2:
        return None
    return pairs[0][0], int(pairs[0][1]), pairs[1][0], int(pairs[1][1])


def unrelaxed_keys() -> set[tuple[str, str, int]]:
    with DATA.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (flagged["dir"], flagged["structure_root"], int(flagged["natoms"]))
            for row in rows
            if (flagged := flagged_row(row)) is not None
        }


def load_rows(exclude_unrelaxed=False) -> pd.DataFrame:
    data = pd.read_csv(DATA)
    data = data[
        (data["structure_root"] == "BCC_SQS")
        & (data["status"] == "OK")
        & data["relax_converged"].astype(str).str.lower().eq("yes")
        & data["natoms"].isin([16, 128])
    ].copy()
    if exclude_unrelaxed:
        flagged = unrelaxed_keys()
        data = data[
            ~data.apply(
                lambda row: (
                    row["dir"],
                    row["structure_root"],
                    int(row["natoms"]),
                )
                in flagged,
                axis=1,
            )
        ].copy()
    parsed = data["dir"].map(parse_dir)
    data["parsed"] = parsed
    data = data[data["parsed"].notna()].copy()
    data["element_a"] = data["parsed"].map(lambda x: x[0])
    data["count_a"] = data["parsed"].map(lambda x: x[1])
    data["element_b"] = data["parsed"].map(lambda x: x[2])
    data["count_b"] = data["parsed"].map(lambda x: x[3])
    data = data[
        (data["count_a"] + data["count_b"] == data["natoms"])
        & (data["count_a"] == data["natoms"] / 2)
        & (data["count_b"] == data["natoms"] / 2)
    ].copy()
    data["pair"] = data.apply(
        lambda row: tuple(sorted((row["element_a"], row["element_b"]))),
        axis=1,
    )
    return data


def size_data(rows: pd.DataFrame, natoms: int):
    subset = rows[rows["natoms"] == natoms]
    pure = {}
    mixed = {}
    for _, row in subset.iterrows():
        a, b = row["pair"]
        volume = float(row["volume_A3"]) / natoms
        if a == b:
            pure[a] = volume
        else:
            mixed[row["pair"]] = volume
    omega = {}
    for pair, volume in mixed.items():
        if all(element in pure for element in pair):
            reference = 0.5 * (pure[pair[0]] + pure[pair[1]])
            omega[pair] = (volume - reference) / reference
    return pure, omega


def percent(value: float) -> float:
    return 100.0 * value


def comparison_data(rows):
    pure16, omega16 = size_data(rows, 16)
    pure128, omega128 = size_data(rows, 128)

    pure_common = sorted(
        element
        for element in set(pure16) & set(pure128)
        if element not in CELL_SIZE_EXCLUSIONS
    )
    pure_delta = {
        element: {
            "volume_16_A3_per_atom": pure16[element],
            "volume_128_A3_per_atom": pure128[element],
            "delta_volume_A3_per_atom": pure128[element] - pure16[element],
            "abs_delta_percent": percent(
                abs(pure128[element] - pure16[element]) / pure16[element]
            ),
        }
        for element in pure_common
    }
    pure_abs = [item["abs_delta_percent"] for item in pure_delta.values()]
    pure_max = (
        max(
            pure_delta.items(),
            key=lambda item: item[1]["abs_delta_percent"],
        )
        if pure_delta
        else (None, None)
    )

    common_pairs = sorted(
        set(omega16) & set(omega128)
        - {
            pair
            for pair in set(omega16) & set(omega128)
            if set(pair) & CELL_SIZE_EXCLUSIONS
        }
    )
    pair_delta = {}
    for pair in common_pairs:
        value16 = omega16[pair]
        value128 = omega128[pair]
        pair_delta["-".join(pair)] = {
            "omega_sf_16": value16,
            "omega_sf_128": value128,
            "delta_omega_sf": value128 - value16,
            "abs_delta_percent": percent(abs(value128 - value16)),
            "ratio_128_to_16": (
                value128 / value16 if value16 != 0 else None
            ),
            "sign_reversed": bool(value16 * value128 < 0),
        }
    pair_abs = [item["abs_delta_percent"] for item in pair_delta.values()]
    sign_reversed = [
        pair for pair, item in pair_delta.items() if item["sign_reversed"]
    ]
    pair_max = (
        max(
            pair_delta.items(),
            key=lambda item: item[1]["abs_delta_percent"],
        )
        if pair_delta
        else (None, None)
    )

    return {
        "pure_delta": pure_delta,
        "pair_delta": pair_delta,
        "raw_row_counts": {
            "16": int((rows["natoms"] == 16).sum()),
            "128": int((rows["natoms"] == 128).sum()),
        },
        "pure_element_volume_comparison": {
            "n_common_elements": len(pure_common),
            "elements": pure_common,
            "median_abs_delta_percent": (
                float(np.median(pure_abs)) if pure_abs else None
            ),
            "max_abs_delta_percent": (
                pure_max[1]["abs_delta_percent"] if pure_max[1] else None
            ),
            "max_abs_delta_element": pure_max[0],
            "by_element": pure_delta,
        },
        "omega_sf_comparison": {
            "n_common_pairs": len(common_pairs),
            "pairs": list(pair_delta),
            "median_abs_delta_percent": (
                float(np.median(pair_abs)) if pair_abs else None
            ),
            "max_abs_delta_percent": (
                pair_max[1]["abs_delta_percent"] if pair_max[1] else None
            ),
            "max_abs_delta_pair": pair_max[0],
            "n_abs_delta_le_0_2_percent": int(sum(v <= 0.2 for v in pair_abs)),
            "sign_reversed_pairs": sign_reversed,
            "by_pair": pair_delta,
        },
    }


def main():
    rows_including = load_rows()
    rows_excluding = load_rows(exclude_unrelaxed=True)
    including = comparison_data(rows_including)
    excluding = comparison_data(rows_excluding)
    metrics = {
        "data_source": "data/sqs_results.csv",
        "filters": {
            "structure_root": "BCC_SQS",
            "status": "OK",
            "relax_converged": "yes",
            "natoms": [16, 128],
            "compositions": "8:8 for 16 atoms; 64:64 for 128 atoms",
            "omega_reference": "same-cell-size pure endpoints",
            "excluded_unrelaxed_volume_rows": True,
        },
        "excluded_elements": sorted(CELL_SIZE_EXCLUSIONS),
        "raw_row_counts": excluding["raw_row_counts"],
        "pure_element_volume_comparison": excluding[
            "pure_element_volume_comparison"
        ],
        "omega_sf_comparison": excluding["omega_sf_comparison"],
        "including_unrelaxed_rows": {
            "raw_row_counts": including["raw_row_counts"],
            "pure_element_volume_comparison": including[
                "pure_element_volume_comparison"
            ],
            "omega_sf_comparison": including["omega_sf_comparison"],
        },
        "figure": str(FIGURE.relative_to(ROOT)),
    }
    make_figure(excluding["pure_delta"], excluding["pair_delta"])
    METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def make_figure(pure_delta, pair_delta, label_threshold=0.2):
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 22,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 8))

    x_pure = np.array([v["volume_16_A3_per_atom"] for v in pure_delta.values()])
    y_pure = np.array([v["volume_128_A3_per_atom"] for v in pure_delta.values()])
    if len(x_pure):
        axes[0].scatter(x_pure, y_pure, s=80, color="#3366aa")
        lo, hi = min(x_pure.min(), y_pure.min()), max(
            x_pure.max(), y_pure.max()
        )
        axes[0].plot(
            [lo, hi], [lo, hi], "--", color="black", linewidth=1.5
        )
    if "Ca" in pure_delta:
        ca = pure_delta["Ca"]
        axes[0].annotate(
            "Ca",
            (ca["volume_16_A3_per_atom"], ca["volume_128_A3_per_atom"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=14,
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85},
        )
    axes[0].set_title("(a) 純元素体積")
    axes[0].set_xlabel(r"16原子 $V$ ($\mathrm{\AA}^3$/atom)")
    axes[0].set_ylabel(r"128原子 $V$ ($\mathrm{\AA}^3$/atom)")
    axes[0].grid(alpha=0.25)

    x_omega = 100.0 * np.array(
        [v["omega_sf_16"] for v in pair_delta.values()]
    )
    y_omega = 100.0 * np.array(
        [v["omega_sf_128"] for v in pair_delta.values()]
    )
    if len(x_omega):
        axes[1].scatter(x_omega, y_omega, s=90, color="#cc5533")
        lo = min(x_omega.min(), y_omega.min())
        hi = max(x_omega.max(), y_omega.max())
        margin = 0.08 * (hi - lo if hi > lo else 1.0)
        axes[1].plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "--",
            color="black",
            linewidth=1.5,
        )
    label_offsets = {
        "Al-Mo": (36, -28),
        "Be-Co": (10, 10),
        "Be-Fe": (-58, -28),
    }
    default_offsets = ((8, 8), (-48, 8), (8, -28), (-48, -28))
    labeled_count = 0
    for pair, values in pair_delta.items():
        if (
            values["abs_delta_percent"] > label_threshold
            or values["sign_reversed"]
        ):
            offset = label_offsets.get(
                pair, default_offsets[labeled_count % len(default_offsets)]
            )
            axes[1].annotate(
                pair,
                (
                    100.0 * values["omega_sf_16"],
                    100.0 * values["omega_sf_128"],
                ),
                xytext=offset,
                textcoords="offset points",
                fontsize=14,
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85},
            )
            labeled_count += 1
    axes[1].set_title(r"(b) $\Omega_\mathrm{sf}$")
    axes[1].set_xlabel(r"16原子 $\Omega_\mathrm{sf}$ (%)")
    axes[1].set_ylabel(r"128原子 $\Omega_\mathrm{sf}$ (%)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
