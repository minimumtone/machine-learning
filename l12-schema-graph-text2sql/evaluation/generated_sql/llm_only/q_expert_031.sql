SELECT entry_id, formula, reduced_formula, formation_enthalpy_ev_per_atom, energy_above_hull, reference_set, prototype, strukturbericht
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND regexp_replace(strukturbericht, '[^A-Za-z0-9]', '', 'g') ILIKE 'L12'
ORDER BY reduced_formula, entry_id;
