SELECT m.formula,
       s.lattice_a,
       ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       cp_sm.value AS shear_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
  AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.formation_energy_per_atom < (
      SELECT ps2.formation_energy_per_atom
      FROM material_entry m2
      JOIN phase_stability ps2 ON ps2.entry_id = m2.entry_id
      WHERE m2.formula = 'Ni3Al'
      ORDER BY ps2.formation_energy_per_atom ASC
      LIMIT 1
  )
  AND cp_bm.value > (
      SELECT cp2.value
      FROM material_entry m2
      JOIN calculation calc2 ON calc2.entry_id = m2.entry_id AND calc2.calculation_type = 'relaxation'
      JOIN calculated_property cp2 ON cp2.calculation_id = calc2.calculation_id
      WHERE m2.formula = 'Ni3Al' AND cp2.property_name = 'bulk_modulus'
      ORDER BY cp2.value DESC
      LIMIT 1
  )
ORDER BY ps.formation_energy_per_atom ASC, cp_bm.value DESC
LIMIT 10000;