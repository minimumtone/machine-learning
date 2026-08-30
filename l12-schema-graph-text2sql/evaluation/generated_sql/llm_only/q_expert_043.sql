SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  MAX(et.shear_modulus_vrh) AS shear_modulus_vrh
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE et.shear_modulus_vrh IS NOT NULL
  AND (
    UPPER(REPLACE(COALESCE(s.strukturbericht, ''), '_', '')) = 'L12'
    OR UPPER(REPLACE(COALESCE(pd.strukturbericht, ''), '_', '')) = 'L12'
  )
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula
ORDER BY
  shear_modulus_vrh DESC
LIMIT 3;
