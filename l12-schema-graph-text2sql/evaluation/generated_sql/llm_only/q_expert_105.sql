SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht,
    ps.energy_above_hull,
    se.miller_index,
    se.surface_energy_j_m2,
    et.bulk_modulus_vrh
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN surface_energy AS se
    ON se.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE ps.is_stable = TRUE
  AND (
      s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1₂%'
  )
  AND se.surface_energy_j_m2 <= 2.0
  AND et.bulk_modulus_vrh >= 180.0;
