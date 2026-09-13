#!/usr/bin/env python3
"""Reproducible metrics for the paper's validation-element scope.

This script deliberately delegates all Omega_sf, q, and lattice-constant
calculations to the repository's existing single-source functions.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))

from chen2023_lattice_screening import load_stable  # noqa: E402
from generate_all_figures import (  # noqa: E402
    EXCLUDE_ELEMENTS,
    compute_omega_sf_pairwise,
    load_compounds,
    load_sqs_data,
    optimize_gamma,
)
from hea_lattice_xgboost import (  # noqa: E402
    ALONSO_TABLE2,
    INDEPENDENT_TEST,
    compute_eq10_scaled,
    compute_vegard,
)


PRACTICAL_ELEMENTS = {
    "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Hf", "Ta", "W",
}
SCOPE_EXCLUDED_ELEMENTS = {"Si"}


def element_set(hea):
    return set(hea["comp"])


def pair_set(db):
    return set(db)


def all_pairs(elements):
    return {
        tuple(sorted(pair))
        for pair in itertools.combinations(sorted(elements), 2)
    }


def filtered_db(db, elements):
    return {pair: value for pair, value in db.items() if set(pair) <= elements}


def rmse(rows, struct, omega, gamma):
    values = [
        compute_eq10_scaled(
            hea["comp"], struct, omega, gamma
        ) - hea["a_exp"]
        for hea in rows
        if hea["struct"] == struct
    ]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def rmse_all(rows, bcc_omega, fcc_omega, q_bcc, q_fcc):
    values = []
    for hea in rows:
        omega = bcc_omega if hea["struct"] == "BCC" else fcc_omega
        gamma = q_bcc if hea["struct"] == "BCC" else q_fcc
        values.append(
            compute_eq10_scaled(hea["comp"], hea["struct"], omega, gamma)
            - hea["a_exp"]
        )
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def test_rmse(rows, bcc_omega, fcc_omega, q_bcc, q_fcc):
    return {
        "BCC": rmse(
            rows, "BCC", bcc_omega, q_bcc
        ),
        "FCC": rmse(
            rows, "FCC", fcc_omega, q_fcc
        ),
        "ALL": rmse_all(rows, bcc_omega, fcc_omega, q_bcc, q_fcc),
    }


def chen_scope_metrics(elements, sqs_bcc, l12, q_fcc):
    """Count stable Chen quinaries and their adopted-model corrections."""
    _, stable = load_stable()
    output = {}
    for struct in ("BCC", "FCC"):
        omega = sqs_bcc if struct == "BCC" else l12
        in_scope = 0
        covered = 0
        corrections = []
        for _, row in stable[stable.phase == struct].iterrows():
            row_elements = set(row.elements)
            if not row_elements <= elements:
                continue
            in_scope += 1
            if not all_pairs(row_elements) <= pair_set(omega):
                continue
            covered += 1
            comp = {element: 1.0 / len(row_elements) for element in row_elements}
            q = 1.0 if struct == "BCC" else q_fcc
            a_vegard = compute_vegard(comp, struct)
            a_pred = compute_eq10_scaled(comp, struct, omega, q)
            corrections.append(100.0 * (a_pred - a_vegard) / a_vegard)
        output[struct] = {
            "stable_all_elements_in_scope": in_scope,
            "stable_10_pair_covered": covered,
            "median_abs_correction_percent": (
                float(np.median(np.abs(corrections))) if corrections else None
            ),
            "n_abs_correction_gt_1percent": int(
                sum(abs(value) > 1.0 for value in corrections)
            ),
        }
    return output


def scope_metrics(label, elements, sqs, ob2, ol12):
    calibration = [
        hea for hea in ALONSO_TABLE2 if element_set(hea) <= elements
    ]
    independent = [
        hea for hea in INDEPENDENT_TEST if element_set(hea) <= elements
    ]
    ob2_scope = filtered_db(ob2, elements)
    ol12_scope = filtered_db(ol12, elements)
    sqs_bcc_scope = filtered_db(sqs["omega_dft"], elements)
    sqs_fcc_dft_scope = filtered_db(sqs["fcc_omega_dft"], elements)
    sqs_fcc_king_scope = filtered_db(sqs["fcc_omega_king"], elements)
    q_bcc, q_fcc = optimize_gamma(calibration, ob2_scope, ol12_scope)
    return {
        "label": label,
        "elements": sorted(elements),
        "n_elements": len(elements),
        "sqs_pairs": {
            "BCC": len(sqs_bcc_scope),
            "FCC_DFT_reference": len(sqs_fcc_dft_scope),
            "FCC_King_reference": len(sqs_fcc_king_scope),
        },
        "ordered_pairs": {
            "B2": len(ob2_scope),
            "L1_2": len(ol12_scope),
        },
        "calibration_all_elements_in_scope": {
            "BCC": sum(hea["struct"] == "BCC" for hea in calibration),
            "FCC": sum(hea["struct"] == "FCC" for hea in calibration),
        },
        "independent_test_all_elements_in_scope": {
            "BCC": sum(hea["struct"] == "BCC" for hea in independent),
            "FCC": sum(hea["struct"] == "FCC" for hea in independent),
        },
        "q_optimized": {"BCC": float(q_bcc), "FCC": float(q_fcc)},
        "independent_test_rmse_q1": test_rmse(
            independent, ob2_scope, ol12_scope, 1.0, 1.0
        ),
        "independent_test_rmse_q_optimized": test_rmse(
            independent, ob2_scope, ol12_scope, q_bcc, q_fcc
        ),
        "chen": chen_scope_metrics(elements, sqs["omega_dft"], ol12_scope, q_fcc),
    }


def main():
    sqs = load_sqs_data()
    ob2, ol12 = compute_omega_sf_pairwise(load_compounds())
    data_elements = {
        element
        for hea in ALONSO_TABLE2 + INDEPENDENT_TEST
        for element in element_set(hea)
    } - SCOPE_EXCLUDED_ELEMENTS
    scopes = {
        "all_elements": set().union(
            *(set(pair) for pair in ob2)
        ) - SCOPE_EXCLUDED_ELEMENTS,
        "candidate_A_practical_16": PRACTICAL_ELEMENTS,
        "candidate_B_data_driven_23": data_elements,
    }
    results = {
        label: scope_metrics(label, elements, sqs, ob2, ol12)
        for label, elements in scopes.items()
    }
    q_bcc_all, q_fcc_all = optimize_gamma(ALONSO_TABLE2, ob2, ol12)
    output = {
        "scope_definition": {
            "candidate_A": sorted(PRACTICAL_ELEMENTS),
            "scope_excluded_elements": sorted(SCOPE_EXCLUDED_ELEMENTS),
            "candidate_B_derived_from": [
                "ALONSO_TABLE2",
                "INDEPENDENT_TEST",
            ],
            "candidate_B": sorted(data_elements),
        },
        "exclude_elements": sorted(EXCLUDE_ELEMENTS),
        "full_database_pair_counts": {
            "SQS_BCC": len(sqs["omega_dft"]),
            "SQS_FCC_DFT_reference_raw": len(sqs["fcc_omega_dft"]),
            "SQS_FCC_King_reference_raw": len(sqs["fcc_omega_king"]),
            "B2": len(ob2),
            "L1_2": len(ol12),
        },
        "full_ordered_q": {
            "BCC": float(q_bcc_all),
            "FCC": float(q_fcc_all),
        },
        "results": results,
        "be": {
            "BCC_SQS_pairs": sum("Be" in pair for pair in sqs["omega_dft"]),
            "FCC_SQS_pairs": sum("Be" in pair for pair in sqs["fcc_omega_dft"]),
            "B2_pairs": sum("Be" in pair for pair in ob2),
            "L1_2_pairs": sum("Be" in pair for pair in ol12),
            "ALONSO_BCC_alloys": sum(
                hea["struct"] == "BCC" and "Be" in element_set(hea)
                for hea in ALONSO_TABLE2
            ),
            "ALONSO_FCC_alloys": sum(
                hea["struct"] == "FCC" and "Be" in element_set(hea)
                for hea in ALONSO_TABLE2
            ),
            "INDEPENDENT_BCC_alloys": sum(
                hea["struct"] == "BCC" and "Be" in element_set(hea)
                for hea in INDEPENDENT_TEST
            ),
            "INDEPENDENT_FCC_alloys": sum(
                hea["struct"] == "FCC" and "Be" in element_set(hea)
                for hea in INDEPENDENT_TEST
            ),
        },
    }
    path = PAPER / "element_scope_metrics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
