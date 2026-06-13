-- VH: 点欠陥の情報があるL1₂化合物で安定かつバルクモジュラスが150GPa以上のものを欠陥タイプとともに出して
-- Tables: material_entry, structure, phase_stability, material_defect, defect_type, calculation, calculated_property (7)
SELECT DISTINCT m.formula, dt.defect_name, cp_bm.value AS bulk_modulus,
       ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN material_defect mdf ON mdf.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = mdf.defect_type_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 150
ORDER BY cp_bm.value DESC
LIMIT 10000;
