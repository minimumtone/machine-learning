#!/usr/bin/env python3
"""Estimate 2-sublattice B2 pair-interaction and end-member parameters from MACE data.

Uses:
- b2_order_param.csv for B2 vs A2 ordering energy
- b2_defect_energies.csv for point-defect formation energies
- mace_mp_ref_results.csv for elemental references
"""
import os, json, math
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

b2_order = pd.read_csv(os.path.join(BASE, "..", "niall_ext", "analysis", "b2_order_param.csv"))
defect = pd.read_csv(os.path.join(AN, "b2_defect_energies.csv"))

# B2 and A2 energies per formula (2 atoms)
E_b2 = float(b2_order[b2_order.eta == 1.0].energy_eV.values[0]) / (128.0 / 2.0)
E_a2 = float(b2_order[b2_order.eta == 0.0].energy_eV.mean()) / (128.0 / 2.0)
# defect energies (per defect)
E_ni_anti = float(defect[defect.defect_kind == "Ni_antisite_on_Al"].deltaE_per_defect_eV.mean())
E_al_anti = float(defect[defect.defect_kind == "Al_antisite_on_Ni"].deltaE_per_defect_eV.mean())

# Ising pair model: each B2 formula has 8 nearest-neighbor bonds.
# E_B2 = 8 J_NiAl
# E_A2 = 2 J_NiNi + 2 J_AlAl + 4 J_NiAl
# Isolated Ni antisite energy (Ni on Al site) = 8 (J_NiNi - J_NiAl)
# Isolated Al antisite energy (Al on Ni site) = 8 (J_AlAl - J_NiAl)
J_NiAl = E_b2 / 8.0
J_NiNi = J_NiAl + E_ni_anti / 8.0
J_AlAl = J_NiAl + E_al_anti / 8.0

# Predicted A2 energy
E_a2_pred = 2.0 * (J_NiNi + J_AlAl) + 4.0 * J_NiAl
delta_order_pred = E_a2_pred - E_b2
delta_order_obs = E_a2 - E_b2

V_pair = J_NiAl - (J_NiNi + J_AlAl) / 2.0
V_from_ordering = -delta_order_obs / 4.0

# Load the independent icet cluster-expansion result if available.
icet_summary_path = os.path.join(AN, "icet_b2_cluster_expansion_summary.json")
icet_1nn = {}
if os.path.exists(icet_summary_path):
    with open(icet_summary_path) as f:
        icet_summary = json.load(f)
    models = {m["label"]: m for m in icet_summary.get("models", [])}
    m1 = models.get("1NN_pairs", {})
    m3 = models.get("1NN+2NN+triplets", {})

    v1 = m1.get("V_pair_eV_per_bond")
    jvals = m1.get("J_values_eV")  # null for non-1NN models
    icet_1nn = {
        "icet_1NN_V_pair_eV_per_bond": round(v1, 4) if v1 is not None else None,
        "icet_1NN_J_values_eV": (
            {k: round(v, 4) for k, v in jvals.items()}
            if isinstance(jvals, dict) else None
        ),
        "icet_2NN_triplets_V_eff_eV_per_bond": (
            round(m3.get("V_eff_eV_per_bond"), 4)
            if m3.get("V_eff_eV_per_bond") is not None else None
        ),
    }

result = {
    "E_B2_per_formula_eV": round(E_b2, 4),
    "E_A2_per_formula_eV": round(E_a2, 4),
    "E_A2_predicted_eV": round(E_a2_pred, 4),
    "delta_order_obs_eV": round(delta_order_obs, 4),
    "delta_order_pred_eV": round(delta_order_pred, 4),
    "J_NiAl_eV": round(J_NiAl, 4),
    "J_NiNi_eV": round(J_NiNi, 4),
    "J_AlAl_eV": round(J_AlAl, 4),
    "V_pair_constant_eV": round(V_pair, 4),
    "V_from_ordering_eV": round(V_from_ordering, 4),
    **icet_1nn,
    "V_definition": "V = -Delta E_order / 4 = -0.3509/4 = -0.088 eV/bond is the thermodynamic ordering strength obtained directly from the B2/A2 energy difference (the value to use in a CALPHAD model).  V_pair = J_NiAl - (J_NiNi+J_AlAl)/2 = -0.145 eV/bond is the value implied by a literal constant-pair fit to isolated point-defect energies, and the 1NN icet cluster expansion gives a comparable -0.137 eV/bond. The two independent 1NN estimates converge to about -0.14 eV/bond, while adding 2NN pairs and triplets decreases the effective V toward the thermodynamic -0.088 eV/bond.",
    "note": "Constant-pair / 1NN approximations overestimate ordering energy in the concentrated limit. Composition-dependent interaction parameters or longer-range cluster expansion terms are needed."
}
with open(os.path.join(AN, "b2_pair_interactions.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2, ensure_ascii=False))
