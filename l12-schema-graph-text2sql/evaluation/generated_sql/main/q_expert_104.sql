SELECT
  m.formula,
  cp.value AS bulk_modulus,
  COUNT(DISTINCT mr.reference_id) AS literature_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
JOIN material_reference mr ON mr.entry_id = m.entry_id
JOIN literature_reference lr ON lr.reference_id = mr.reference_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
GROUP BY m.entry_id, m.formula, cp.value
ORDER BY cp.value DESC
LIMIT 10;
