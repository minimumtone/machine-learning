SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh,
  COUNT(DISTINCT lr.reference_id) AS literature_count
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
JOIN material_reference AS mr
  ON mr.entry_id = me.entry_id
JOIN literature_reference AS lr
  ON lr.reference_id = mr.reference_id
WHERE
  et.bulk_modulus_vrh IS NOT NULL
  AND (
    s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR s.prototype ILIKE '%L1_2%'
    OR s.prototype ILIKE '%L12%'
    OR s.prototype ILIKE '%L1₂%'
  )
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula
ORDER BY
  MAX(et.bulk_modulus_vrh) DESC
LIMIT 10;
