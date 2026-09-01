SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    ps.energy_above_hull
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
WHERE ps.is_stable = TRUE
  AND me.number_of_elements > 1
  AND EXISTS (
      SELECT 1
      FROM composition AS c
      WHERE c.entry_id = me.entry_id
        AND c.element = 'Ni'
  )
  AND (
      s.strukturbericht IN ('L12', 'L1_2', 'L1₂')
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L1₂%'
  )
ORDER BY me.reduced_formula, me.entry_id;
