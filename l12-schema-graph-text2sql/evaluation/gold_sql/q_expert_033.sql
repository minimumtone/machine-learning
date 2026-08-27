-- Uses the formation_enthalpy view so the benchmark exercises the view's
-- reference-set-safe derivation instead of re-implementing it by hand.
SELECT fe.entry_id, fe.formula, fe.formation_enthalpy_ev_per_atom,
       fe.energy_above_hull
FROM formation_enthalpy fe
WHERE (fe.prototype = 'B2' OR fe.strukturbericht = 'B2')
  AND fe.energy_above_hull <= 0.001
  AND fe.formation_enthalpy_ev_per_atom < 0
ORDER BY fe.formation_enthalpy_ev_per_atom
LIMIT 10000;
