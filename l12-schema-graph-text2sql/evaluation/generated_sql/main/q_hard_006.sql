SELECT c.element AS a_site,
       AVG(s.lattice_a) AS avg_lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A-site'
GROUP BY c.element
ORDER BY c.element
LIMIT 10000;
