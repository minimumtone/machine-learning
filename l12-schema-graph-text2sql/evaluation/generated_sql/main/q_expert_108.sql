SELECT m.formula,
       ps.is_stable,
       ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       dt.category AS defect_category,
       dt.defect_name,
       md.site,
       md.concentration,
       md.formation_energy
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
     AND cp_bm.property_name = 'bulk_modulus'
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp_bm.value >= 150
ORDER BY cp_bm.value DESC, ps.energy_above_hull ASC
LIMIT 10000;
