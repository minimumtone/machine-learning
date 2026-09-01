SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  MAX(et.bulk_modulus_vrh) AS bulk_modulus_gpa
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE et.bulk_modulus_vrh >= 200
  AND (
    regexp_replace(upper(coalesce(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
  )
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula
ORDER BY
  bulk_modulus_gpa DESC;
