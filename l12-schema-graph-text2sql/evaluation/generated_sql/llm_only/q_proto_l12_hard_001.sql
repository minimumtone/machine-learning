SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom,
  energy_above_hull,
  prototype,
  strukturbericht,
  space_group
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND UPPER(REGEXP_REPLACE(COALESCE(strukturbericht, ''), '[^A-Za-z0-9]', '', 'g')) = 'L12'
ORDER BY formation_enthalpy_ev_per_atom ASC;
