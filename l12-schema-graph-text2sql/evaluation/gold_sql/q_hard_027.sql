SELECT DISTINCT m.formula, ca.element AS a_site, cb.element AS b_site
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND (ca.element = 'Ni' OR ca.element = 'Co')
  AND (cb.element = 'Al' OR cb.element = 'Ti')
ORDER BY m.formula
LIMIT 10000;