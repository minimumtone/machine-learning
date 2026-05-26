#!/usr/bin/env python3
"""
HEA Lattice Constant Prediction Web Application.

Uses the DFT-SS model (Alonso model with structure-specific DFT Ω_sf)
to predict lattice constants for arbitrary n-component HEAs.

Usage:
    python webapp/app.py
    # Open http://localhost:5000 in browser
"""

import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# Load pre-computed model data
DATA_PATH = Path(__file__).parent / "model_data.json"
with open(DATA_PATH) as f:
    MODEL = json.load(f)

KING_VOLUMES = MODEL["king_atomic_volumes"]
OMEGA_B2 = {tuple(k.split(",")): v for k, v in MODEL["omega_sf_b2"].items()}
OMEGA_L12 = {tuple(k.split(",")): v for k, v in MODEL["omega_sf_l12"].items()}
GAMMA_BCC = MODEL["gamma_bcc"]
GAMMA_FCC = MODEL["gamma_fcc"]


def predict_lattice(comp: dict, struct: str) -> dict:
    """
    Predict HEA lattice constant using the Alonso model with DFT Ω_sf.

    Parameters
    ----------
    comp : dict
        Element -> atomic fraction, e.g. {"Co": 0.25, "Cr": 0.25, "Fe": 0.25, "Ni": 0.25}
    struct : str
        "BCC" or "FCC"

    Returns
    -------
    dict with a_vegard, a_predicted, details
    """
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    total = fracs.sum()
    if total <= 0:
        raise ValueError("Composition fractions must be positive")
    fracs = fracs / total

    # Check elements
    missing = [e for e in elements if e not in KING_VOLUMES]
    if missing:
        raise ValueError(f"Unknown elements: {missing}. Available: {sorted(KING_VOLUMES.keys())}")

    vols = np.array([KING_VOLUMES[e] for e in elements])
    n_auc = 4 if struct == "FCC" else 2
    omega_sf = OMEGA_L12 if struct == "FCC" else OMEGA_B2
    gamma = GAMMA_FCC if struct == "FCC" else GAMMA_BCC

    # Vegard's law
    v_vegard = float(np.sum(fracs * vols))
    a_vegard = (n_auc * v_vegard) ** (1 / 3)

    # Ω_sf correction
    n_elem = len(elements)
    correction = 0.0
    pair_details = []
    missing_pairs = []
    for i in range(n_elem):
        for j in range(n_elem):
            if i == j:
                continue
            pair = tuple(sorted([elements[i], elements[j]]))
            omega = omega_sf.get(pair, 0.0)
            contrib = fracs[i] * fracs[j] * vols[j] * omega
            correction += contrib
            if omega != 0.0:
                pair_details.append({
                    "pair": f"{elements[i]}-{elements[j]}",
                    "omega_sf": round(omega, 6),
                    "contribution": round(contrib, 6),
                })
            else:
                if pair not in [tuple(sorted(p)) for p in missing_pairs]:
                    missing_pairs.append(pair)

    v_total = n_auc * (v_vegard + gamma * correction)
    if v_total <= 0:
        a_pred = a_vegard
        warning = "Negative volume; fell back to Vegard"
    else:
        a_pred = v_total ** (1 / 3)
        warning = None

    # Vegard fallback fraction
    vegard_fallback = len(missing_pairs) > 0

    return {
        "a_vegard": round(a_vegard, 4),
        "a_predicted": round(a_pred, 4),
        "structure": struct,
        "n_auc": n_auc,
        "gamma": round(gamma, 4),
        "v_vegard_per_atom": round(v_vegard, 4),
        "v_total_per_cell": round(v_total, 4) if v_total > 0 else None,
        "correction_term": round(gamma * correction, 6),
        "pair_contributions": pair_details,
        "missing_pairs": [f"{a}-{b}" for a, b in missing_pairs],
        "vegard_fallback": vegard_fallback,
        "warning": warning,
        "composition": {e: round(float(fracs[i]), 4) for i, e in enumerate(elements)},
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body: {"composition": {"Co": 0.25, "Cr": 0.25, ...}, "structure": "FCC"}
    """
    try:
        data = request.get_json()
        comp = data.get("composition", {})
        struct = data.get("structure", "BCC").upper()

        if struct not in ("BCC", "FCC"):
            return jsonify({"error": "structure must be 'BCC' or 'FCC'"}), 400

        if not comp:
            return jsonify({"error": "composition is empty"}), 400

        # Convert string values to float
        comp = {k: float(v) for k, v in comp.items() if float(v) > 0}

        result = predict_lattice(comp, struct)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/api/elements", methods=["GET"])
def api_elements():
    """Return list of available elements with their atomic volumes."""
    return jsonify({
        "elements": sorted(KING_VOLUMES.keys()),
        "atomic_volumes": KING_VOLUMES,
        "gamma_bcc": GAMMA_BCC,
        "gamma_fcc": GAMMA_FCC,
        "n_b2_pairs": len(OMEGA_B2),
        "n_l12_pairs": len(OMEGA_L12),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting HEA Lattice Prediction server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
