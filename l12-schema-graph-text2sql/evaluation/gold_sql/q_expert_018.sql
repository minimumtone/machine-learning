SELECT s.prototype, COUNT(DISTINCT m.entry_id) AS cnt
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Mn'
GROUP BY s.prototype
ORDER BY cnt DESC
LIMIT 10000;
