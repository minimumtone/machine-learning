SELECT m.entry_id, m.formula, et.poisson_ratio, et.bulk_modulus_vrh, et.shear_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.poisson_ratio < 0.25
ORDER BY et.poisson_ratio
LIMIT 10000;
