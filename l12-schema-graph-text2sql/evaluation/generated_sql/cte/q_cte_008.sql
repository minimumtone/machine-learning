SELECT c.element AS b_site,
       AVG(ps.formation_energy_per_atom) AS avg_rebased_formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.site_label = 'B-site'
GROUP BY c.element
HAVING AVG(ps.formation_energy_per_atom) < -0.4
ORDER BY avg_rebased_formation_energy_per_atom ASC
LIMIT 10000;
