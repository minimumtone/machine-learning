SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, et.shear_modulus_vrh,
  ROUND((et.bulk_modulus_vrh / NULLIF(et.shear_modulus_vrh, 0))::numeric, 3) AS bg_ratio
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.shear_modulus_vrh > 0
  AND (et.bulk_modulus_vrh / et.shear_modulus_vrh) >= 2.0
ORDER BY bg_ratio DESC
LIMIT 10000;
