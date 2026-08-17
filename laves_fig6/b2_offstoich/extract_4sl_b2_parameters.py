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
    "V_definition": "V = -Delta E_order / 4 = -0.3509/4 = -0.088 eV/bond is the thermodynamic ordering strength obtained directly from the B2/A2 energy difference.  V_pair = J_NiAl - (J_NiNi+J_AlAl)/2 = -0.145 eV/bond is the value implied by a literal constant-pair fit to the *isolated* point-defect energies; the 65% disagreement shows the constant-pair approximation breaks down for concentrated NiAl.",
    "note": "Constant pair model overestimates ordering energy (concentrated limit). Composition-dependent interaction parameters are needed."
}
with open(os.path.join(AN, "b2_pair_interactions.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2, ensure_ascii=False))
