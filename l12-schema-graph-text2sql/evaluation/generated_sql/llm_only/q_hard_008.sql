SELECT DISTINCT
  me.entry_id,
  me.formula,
  et.bulk_modulus_vrh AS bulk_modulus_gpa
FROM material_entry AS me
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
LEFT JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
LEFT JOIN formation_enthalpy AS fh
  ON fh.entry_id = me.entry_id
WHERE et.bulk_modulus_vrh >= 180
  AND (
    s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR fh.strukturbericht IN ('L1_2', 'L12', 'L1₂')
  )
ORDER BY et.bulk_modulus_vrh DESC;
