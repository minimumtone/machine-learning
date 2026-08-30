SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    ps.energy_above_hull,
    s.prototype,
    s.strukturbericht,
    et.bulk_modulus_vrh AS bulk_modulus_gpa
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE ps.is_stable = TRUE
  AND et.bulk_modulus_vrh >= 160
  AND (
      s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1₂%'
  )
ORDER BY et.bulk_modulus_vrh DESC;
