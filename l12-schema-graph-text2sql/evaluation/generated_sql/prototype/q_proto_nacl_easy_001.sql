SELECT m.formula, s.prototype, s.strukturbericht, s.space_group_number
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'NaCl'
   OR s.strukturbericht = 'B1'
ORDER BY m.formula
LIMIT 10000;
