SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom,
  energy_above_hull,
  prototype,
  strukturbericht
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND upper(regexp_replace(replace(strukturbericht, '₂', '2'), '[^A-Z0-9]', '', 'g')) = 'L12'
ORDER BY formation_enthalpy_ev_per_atom ASC;
