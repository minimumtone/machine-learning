SELECT c.element AS a_site_element,
       AVG(ps.formation_energy_per_atom) AS avg_eform,
       COUNT(DISTINCT m.entry_id) AS count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND c.site_label = 'A'
GROUP BY c.element
ORDER BY avg_eform ASC
LIMIT 100;