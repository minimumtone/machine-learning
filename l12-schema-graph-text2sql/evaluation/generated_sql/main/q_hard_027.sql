SELECT DISTINCT
  m.formula,
  c_a.element AS a_site,
  c_a.atomic_fraction AS a_site_atomic_fraction,
  c_b.element AS b_site,
  c_b.atomic_fraction AS b_site_atomic_fraction
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c_a ON c_a.entry_id = m.entry_id
JOIN composition c_b ON c_b.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c_a.site_label = 'A-site'
  AND c_a.element IN ('Ni', 'Co')
  AND c_b.site_label = 'B-site'
  AND c_b.element IN ('Al', 'Ti')
ORDER BY m.formula, c_a.element, c_b.element
LIMIT 10000;
