SELECT COUNT(DISTINCT m.entry_id) AS l12_compound_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
LIMIT 10000;
