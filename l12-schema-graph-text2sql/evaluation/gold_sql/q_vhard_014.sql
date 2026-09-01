SELECT
  CASE WHEN e.period_number = 4 AND e.block = 'd' THEN '3d'
       ELSE '4d/5d' END AS a_site_series,
  COUNT(*) AS n_compounds,
  AVG(ps.energy_above_hull) AS avg_energy_above_hull,
  AVG(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1.0 ELSE 0.0 END) AS stable_fraction,
  AVG(ps.formation_energy_per_atom) AS avg_formation_energy
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A-site'
JOIN element e ON e.symbol = ca.element
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND e.block = 'd'
  AND e.category = 'transition_metal'
  AND e.period_number IN (4, 5, 6)
GROUP BY 1
ORDER BY a_site_series ASC
LIMIT 10000;
