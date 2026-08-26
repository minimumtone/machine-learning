-- VH: 安定なL1₂化合物で表面エネルギーが2.0 J/m²以下かつバルクモジュラスが180GPa以上のものを全て出して
-- Tables: material_entry, structure, phase_stability, surface_energy, calculation, calculated_property (6)
SELECT m.formula, se.surface_energy_j_m2, cp_bm.value AS bulk_modulus,
       ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND se.surface_energy_j_m2 <= 2.0
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 180
ORDER BY se.surface_energy_j_m2 ASC
LIMIT 10000;
