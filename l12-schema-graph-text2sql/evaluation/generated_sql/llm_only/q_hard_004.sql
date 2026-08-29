SELECT
  me.entry_id,
  me.formula,
  et.bulk_modulus_vrh AS bulk_modulus
FROM material_entry AS me
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE et.bulk_modulus_vrh IS NOT NULL
  AND EXISTS (
    SELECT 1
    FROM phase_stability AS ps
    WHERE ps.entry_id = me.entry_id
      AND ps.is_stable = TRUE
  )
  AND EXISTS (
    SELECT 1
    FROM structure AS s
    LEFT JOIN prototype_definition AS pd
      ON pd.prototype_id = s.prototype
    WHERE s.entry_id = me.entry_id
      AND (
        s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
        OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      )
  )
ORDER BY et.bulk_modulus_vrh DESC;
