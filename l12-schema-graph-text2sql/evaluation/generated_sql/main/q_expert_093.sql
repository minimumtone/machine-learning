SELECT c.element AS a_site, COUNT(DISTINCT m.entry_id) AS l12_compound_count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A-site'
GROUP BY c.element
ORDER BY l12_compound_count DESC
LIMIT 1;
