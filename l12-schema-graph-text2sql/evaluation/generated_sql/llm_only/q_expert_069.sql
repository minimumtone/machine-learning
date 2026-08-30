SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ARRAY_AGG(DISTINCT se.surface_energy_j_m2) FILTER (
    WHERE REGEXP_REPLACE(se.miller_index, '[^0-9-]', '', 'g') = '100'
  ) AS surface_100_energies_j_m2,
  ARRAY_AGG(DISTINCT se.surface_energy_j_m2) FILTER (
    WHERE REGEXP_REPLACE(se.miller_index, '[^0-9-]', '', 'g') = '110'
  ) AS surface_110_energies_j_m2
FROM material_entry me
JOIN surface_energy se
  ON se.entry_id = me.entry_id
WHERE REGEXP_REPLACE(se.miller_index, '[^0-9-]', '', 'g') IN ('100', '110')
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula
HAVING
  COUNT(*) FILTER (
    WHERE REGEXP_REPLACE(se.miller_index, '[^0-9-]', '', 'g') = '100'
  ) > 0
  AND
  COUNT(*) FILTER (
    WHERE REGEXP_REPLACE(se.miller_index, '[^0-9-]', '', 'g') = '110'
  ) > 0
ORDER BY me.entry_id;
