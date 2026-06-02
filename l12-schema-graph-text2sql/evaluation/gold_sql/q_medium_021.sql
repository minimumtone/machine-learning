SELECT c.element AS a_site_element, COUNT(DISTINCT m.entry_id) AS compound_count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A'
GROUP BY c.element
ORDER BY compound_count DESC
LIMIT 100;