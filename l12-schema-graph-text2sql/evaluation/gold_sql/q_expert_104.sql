-- VH: 文献参照があるL1₂化合物のうち、バルクモジュラスが最も高い上位10化合物を文献数とともに出して
-- Tables: material_entry, structure, material_reference, calculation, calculated_property (5)
SELECT m.formula, cp_bm.value AS bulk_modulus,
       COUNT(DISTINCT mr.reference_id) AS ref_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_reference mr ON mr.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
GROUP BY m.formula, cp_bm.value
HAVING COUNT(DISTINCT mr.reference_id) >= 1
ORDER BY cp_bm.value DESC
LIMIT 10;
