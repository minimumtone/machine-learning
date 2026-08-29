SELECT DISTINCT
  me.entry_id,
  me.formula,
  et.shear_modulus_vrh AS shear_modulus
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
   OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
ORDER BY me.formula;
