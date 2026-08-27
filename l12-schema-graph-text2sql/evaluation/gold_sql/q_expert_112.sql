-- VH: 実験的に合成されたL1₂化合物のうち安定でバルクモジュラスが150GPa以上のものを合成方法・温度とともに出して
-- Tables: material_entry, structure, phase_stability, material_synthesis, synthesis_method, calculation, calculated_property (7)
SELECT m.formula, sm.method_name, msyn.temperature_k,
       cp_bm.value AS bulk_modulus, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN material_synthesis msyn ON msyn.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = msyn.synthesis_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 150
ORDER BY cp_bm.value DESC
LIMIT 10000;
