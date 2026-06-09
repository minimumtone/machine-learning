SELECT m.entry_id, m.formula, tp.debye_temperature_k
FROM material_entry m
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE tp.debye_temperature_k >= 500
ORDER BY tp.debye_temperature_k DESC
LIMIT 10000;
