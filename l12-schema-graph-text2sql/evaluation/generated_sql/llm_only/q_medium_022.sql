SELECT
  entry_id,
  formula,
  reduced_formula,
  prototype,
  strukturbericht,
  formation_enthalpy_ev_per_atom,
  reference_set
FROM formation_enthalpy
WHERE formation_enthalpy_ev_per_atom IS NOT NULL
  AND LOWER(REPLACE(REPLACE(COALESCE(strukturbericht, ''), '₂', '2'), '_', '')) = 'l12'
ORDER BY formation_enthalpy_ev_per_atom ASC
LIMIT 5;
