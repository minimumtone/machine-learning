SELECT DISTINCT
    me.formula,
    mp.total_magnetization,
    et.bulk_modulus_vrh
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN magnetic_property AS mp
    ON mp.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE ps.is_stable = TRUE
  AND mp.total_magnetization IS NOT NULL
  AND ABS(mp.total_magnetization) > 0
  AND et.bulk_modulus_vrh IS NOT NULL
  AND (
      s.strukturbericht IN ('L12', 'L1_2', 'L1₂')
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L1₂%'
  )
ORDER BY me.formula;
