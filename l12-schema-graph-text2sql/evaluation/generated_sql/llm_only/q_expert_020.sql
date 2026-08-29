SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    ps.energy_above_hull,
    ps.reference_set
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
WHERE ps.is_stable = TRUE
  AND EXISTS (
      SELECT 1
      FROM composition AS c
      WHERE c.entry_id = me.entry_id
        AND c.element = 'V'
  )
  AND EXISTS (
      SELECT 1
      FROM composition AS c
      WHERE c.entry_id = me.entry_id
        AND c.element = 'Al'
  )
ORDER BY
    ps.energy_above_hull,
    me.reduced_formula,
    me.entry_id;
