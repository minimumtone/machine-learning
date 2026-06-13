-- VH: Ni基合金系に属するL1₂化合物で、安定かつバルクモジュラスが最も高いものを合金系名とともにトップ10出して
-- Tables: material_entry, structure, phase_stability, material_alloy_system, alloy_system, calculation, calculated_property (7)
SELECT m.formula, als.system_name, cp_bm.value AS bulk_modulus,
       ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN material_alloy_system mas ON mas.entry_id = m.entry_id
JOIN alloy_system als ON als.alloy_system_id = mas.alloy_system_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND als.system_name LIKE '%Ni%'
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY cp_bm.value DESC
LIMIT 10;
