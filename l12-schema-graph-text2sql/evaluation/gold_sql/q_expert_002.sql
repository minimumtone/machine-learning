SELECT m.entry_id, m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'B2' OR s.strukturbericht = 'B2'
ORDER BY m.formula
LIMIT 10000;
