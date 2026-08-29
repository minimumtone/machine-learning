SELECT c.element AS a_site,
       AVG(ps.formation_energy_per_atom) AS avg_rereferenced_formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND c.site_label = 'A-site'
GROUP BY c.element
ORDER BY avg_rereferenced_formation_energy_per_atom ASC
LIMIT 10000;
