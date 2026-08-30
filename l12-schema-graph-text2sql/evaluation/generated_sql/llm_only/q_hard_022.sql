SELECT DISTINCT
  me.entry_id,
  me.formula,
  c.calculation_id,
  et.bulk_modulus_vrh AS bulk_modulus_gpa
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE c.functional = 'PBE'
  AND et.bulk_modulus_vrh IS NOT NULL
  AND (
    s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR s.prototype ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L12%'
    OR pd.prototype_name ILIKE '%L1₂%'
  )
ORDER BY me.formula, me.entry_id;
