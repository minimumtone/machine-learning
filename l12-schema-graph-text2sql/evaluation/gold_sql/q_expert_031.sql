-- Uses the formation_enthalpy view so the benchmark exercises the view's
-- reference-set-safe derivation instead of re-implementing it by hand.
SELECT fe.entry_id, fe.formula, fe.energy_above_hull,
       fe.formation_enthalpy_ev_per_atom
FROM formation_enthalpy fe
WHERE (fe.prototype = 'L12' OR fe.strukturbericht = 'L12')
  AND fe.energy_above_hull <= 0.001
ORDER BY fe.formation_enthalpy_ev_per_atom
LIMIT 10000;
