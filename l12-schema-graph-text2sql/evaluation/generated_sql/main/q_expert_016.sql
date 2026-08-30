SELECT DISTINCT m.formula, c.element, c.site_label
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'B-site'
  AND c.element IN ('Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd')
ORDER BY c.element, m.formula
LIMIT 10000;
