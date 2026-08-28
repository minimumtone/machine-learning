SELECT m.entry_id, m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'BiF3' OR s.strukturbericht = 'D03'
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
