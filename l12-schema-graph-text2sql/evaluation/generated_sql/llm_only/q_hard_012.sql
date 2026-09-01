SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
  energy_above_hull,
  prototype,
  strukturbericht,
  reference_set
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND regexp_replace(
        upper(replace(replace(COALESCE(strukturbericht, ''), '₂', '2'), '_', '')),
        '[^A-Z0-9]',
        '',
        'g'
      ) = 'L12'
ORDER BY formation_enthalpy_ev_per_atom ASC NULLS LAST
LIMIT 10;
