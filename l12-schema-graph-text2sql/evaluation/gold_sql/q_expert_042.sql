SELECT m.entry_id, m.formula, et.bulk_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.bulk_modulus_vrh >= 200
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;
