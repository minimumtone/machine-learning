-- VH: NiとAlを両方含むL1₂化合物で、Ni3Alの格子定数(3.57Å)との差が0.1Å以内かつバルクモジュラスが100GPa以上のものを出して
-- Tables: material_entry, composition(×2 self-join), structure, calculation, calculated_property (5 distinct)
SELECT m.formula, s.lattice_a, cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN composition c_ni ON c_ni.entry_id = m.entry_id AND c_ni.element = 'Ni'
JOIN composition c_al ON c_al.entry_id = m.entry_id AND c_al.element = 'Al'
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ABS(s.lattice_a - 3.57) <= 0.1
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 100
ORDER BY ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;
