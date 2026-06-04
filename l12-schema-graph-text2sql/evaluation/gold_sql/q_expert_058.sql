SELECT m.entry_id, m.formula, tp.gruneisen_parameter
FROM material_entry m
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE tp.gruneisen_parameter >= 2.0
ORDER BY tp.gruneisen_parameter DESC
LIMIT 10000;
