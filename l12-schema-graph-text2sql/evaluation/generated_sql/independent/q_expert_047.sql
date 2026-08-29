SELECT
  m.formula,
  et.is_stable
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = calc.calculation_id
WHERE
  et.is_stable = FALSE
  AND (calc.method = 'DFT' OR calc.calculation_type = 'DFT')
ORDER BY m.formula ASC
LIMIT 10000;
