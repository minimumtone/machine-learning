SELECT m.formula, calc.functional
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
WHERE calc.functional IS NOT NULL
  AND calc.functional <> 'GGA-PBE'
ORDER BY calc.functional, m.formula
LIMIT 10000;
