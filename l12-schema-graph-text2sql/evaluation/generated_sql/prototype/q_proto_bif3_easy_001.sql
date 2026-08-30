SELECT m.formula, s.prototype
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'BiF3'
ORDER BY m.formula
LIMIT 10000;
