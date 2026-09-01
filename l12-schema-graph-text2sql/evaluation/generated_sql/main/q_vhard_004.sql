SELECT m.formula, ps.is_stable, ps.energy_above_hull, cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
     AND cp_bm.property_name = 'bulk_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND (
    EXISTS (
      SELECT 1
      FROM composition c_ni
      WHERE c_ni.entry_id = m.entry_id
        AND c_ni.element = 'Ni'
    )
    OR EXISTS (
      SELECT 1
      FROM composition c_co
      WHERE c_co.entry_id = m.entry_id
        AND c_co.element = 'Co'
    )
  )
ORDER BY cp_bm.value DESC
LIMIT 10;
