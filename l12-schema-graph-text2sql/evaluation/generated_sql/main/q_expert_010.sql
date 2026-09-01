SELECT m.formula, s.prototype, s.strukturbericht, s.space_group_number
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'BiF3' OR s.strukturbericht = 'BiF3'
ORDER BY m.formula
LIMIT 10000;
