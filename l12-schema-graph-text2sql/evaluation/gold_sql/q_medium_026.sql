SELECT c.element AS b_site_element,
       AVG(ps.formation_energy_per_atom) AS avg_formation_energy
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'B-site'
GROUP BY c.element
ORDER BY avg_formation_energy ASC
LIMIT 10000;