SELECT DISTINCT m.source_db, calc.calculation_type, calc.method, calc.functional
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
ORDER BY m.source_db, calc.calculation_type, calc.method, calc.functional
LIMIT 10000;
