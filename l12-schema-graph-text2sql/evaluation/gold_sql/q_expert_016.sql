SELECT m.entry_id, m.formula, c.element, c.atomic_fraction
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN element e ON e.symbol = c.element
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.site_label = 'B-site'
  AND e.category = 'transition_metal'
  AND e.block = 'd'
  AND e.period_number = 5
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
