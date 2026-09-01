SELECT m.formula, s.prototype, s.strukturbericht, s.space_group_number
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'BCC_B2'
   OR s.prototype = 'B2'
   OR s.strukturbericht = 'B2'
   OR s.formula_type = 'BCC_B2'
ORDER BY m.formula
LIMIT 10000;
