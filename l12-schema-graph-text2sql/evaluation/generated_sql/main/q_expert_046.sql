SELECT m.formula,
       cp_bm.value AS bulk_modulus,
       cp_sm.value AS shear_modulus,
       cp_bm.value / NULLIF(cp_sm.value, 0) AS b_g_ratio
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
     AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
     AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.value / NULLIF(cp_sm.value, 0) >= 2
ORDER BY b_g_ratio DESC
LIMIT 10000;
