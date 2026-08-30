SELECT m.formula, cp_bm.value AS bulk_modulus, cp_hull.value AS energy_above_hull
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
JOIN calculated_property cp_hull ON cp_hull.calculation_id = calc.calculation_id
WHERE cp_bm.property_name = 'bulk_modulus'
  AND cp_hull.property_name = 'energy_above_hull'
  AND cp_hull.value <= 0.01
  AND cp_bm.value > (
      SELECT cp_ref.value
      FROM material_entry m_ref
      JOIN calculation calc_ref ON calc_ref.entry_id = m_ref.entry_id
      JOIN calculated_property cp_ref ON cp_ref.calculation_id = calc_ref.calculation_id
      WHERE cp_ref.property_name = 'bulk_modulus'
        AND (m_ref.formula = 'Ni3Al' OR m_ref.reduced_formula = 'Ni3Al')
      ORDER BY cp_ref.value DESC
      LIMIT 1
  )
ORDER BY cp_bm.value DESC
LIMIT 10000;
