SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  c.calculation_id,
  c.method,
  c.functional,
  et.bulk_modulus_vrh,
  et.shear_modulus_vrh,
  et.youngs_modulus,
  et.poisson_ratio
FROM elastic_tensor et
JOIN calculation c
  ON et.calculation_id = c.calculation_id
JOIN material_entry me
  ON c.entry_id = me.entry_id
WHERE et.is_stable = FALSE
  AND c.method ILIKE '%DFT%'
ORDER BY me.reduced_formula, me.entry_id;
