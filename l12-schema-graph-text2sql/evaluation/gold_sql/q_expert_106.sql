-- VH: 磁性を持つ安定なL1₂化合物について、磁化とバルクモジュラスの関係を化学式とともに出して
-- Tables: material_entry, structure, phase_stability, magnetic_property, calculation, calculated_property (6)
SELECT m.formula, mp.total_magnetization, cp_bm.value AS bulk_modulus,
       ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND mp.total_magnetization > 0
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY mp.total_magnetization DESC
LIMIT 10000;
