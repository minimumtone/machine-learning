-- VH: 安定なL1₂化合物のうち、バルクモジュラスが180GPa以上で格子定数が3.9Å以下のものを
-- 化学式・格子定数・バルクモジュラス・形成エネルギーとともに一覧して
-- Tables: material_entry, structure, phase_stability, calculation, calculated_property (5)
SELECT DISTINCT m.formula, s.lattice_a, cp_bm.value AS bulk_modulus,
       ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 180
  AND s.lattice_a <= 3.9
ORDER BY cp_bm.value DESC
LIMIT 10000;
