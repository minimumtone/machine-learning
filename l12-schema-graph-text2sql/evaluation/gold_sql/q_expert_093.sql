SELECT c.element, COUNT(DISTINCT m.entry_id) AS cnt
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.atomic_fraction >= 0.70
GROUP BY c.element
ORDER BY cnt DESC
LIMIT 10000;
