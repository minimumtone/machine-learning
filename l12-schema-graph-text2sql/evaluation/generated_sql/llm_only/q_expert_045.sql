SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  et.poisson_ratio
FROM material_entry AS me
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE et.poisson_ratio >= 0.3
  AND me.number_of_elements > 1
ORDER BY et.poisson_ratio DESC, me.formula;
