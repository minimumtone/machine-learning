-- VH: 安定なL1₂化合物でバルクモジュラスとせん断弾性率の比(B/G比)が2.0以上かつ格子定数が4.0Å以下のものをランキングして
-- Tables: material_entry, structure, phase_stability, calculation, calculated_property(×2 self-join) (5 distinct)
SELECT m.formula, s.lattice_a,
       cp_bm.value AS bulk_modulus,
       cp_sm.value AS shear_modulus,
       cp_bm.value / cp_sm.value AS bg_ratio
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
  AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND s.lattice_a <= 4.0
  AND cp_sm.value > 0
  AND cp_bm.value / cp_sm.value >= 2.0
ORDER BY cp_bm.value / cp_sm.value DESC
LIMIT 10000;
