SELECT ca.element AS a_site,
       cb.element AS b_site,
       COUNT(DISTINCT m.entry_id) AS l12_compound_count,
       COUNT(DISTINCT m.entry_id) FILTER (WHERE ps.is_stable = TRUE) AS count,
       COUNT(DISTINCT m.entry_id) FILTER (WHERE ps.is_stable = TRUE) * 100.0 / COUNT(DISTINCT m.entry_id) AS stable_percentage,
       AVG(ps.energy_above_hull) AS avg_energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition ca ON ca.entry_id = m.entry_id
JOIN composition cb ON cb.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ca.site_label = 'A-site'
  AND cb.site_label = 'B-site'
GROUP BY ca.element, cb.element
ORDER BY l12_compound_count DESC, stable_percentage DESC
LIMIT 10000;
