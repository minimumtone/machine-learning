SELECT COUNT(*) AS l12_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12'
LIMIT 100;