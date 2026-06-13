-- VH: 安定なL1₂化合物の弾性テンソルのVRHバルクモジュラスが200GPa以上かつ格子定数が3.5-4.0Åの範囲にあるものを出して
-- Tables: material_entry, structure, phase_stability, elastic_tensor, calculation (5)
SELECT DISTINCT m.formula, s.lattice_a, et.bulk_modulus_vrh, et.shear_modulus_vrh,
       et.youngs_modulus, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND et.bulk_modulus_vrh >= 200
  AND s.lattice_a BETWEEN 3.5 AND 4.0
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;
