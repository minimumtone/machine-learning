SELECT c_a.element AS a_site,
       c_b.element AS b_site,
       COUNT(*) AS l12_combination_count,
       AVG(ps.energy_above_hull) AS avg_energy_above_hull,
       COUNT(*) FILTER (WHERE ps.energy_above_hull <= 0.001) AS count,
       COUNT(*) FILTER (WHERE ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05) AS metastable_count,
       COUNT(*) FILTER (WHERE ps.energy_above_hull > 0.05) AS unstable_count,
       COUNT(*) FILTER (WHERE ps.energy_above_hull <= 0.001) * 100.0 / COUNT(*) AS stable_percentage
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c_a ON c_a.entry_id = m.entry_id
JOIN composition c_b ON c_b.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c_a.site_label = 'A-site'
  AND c_b.site_label = 'B-site'
GROUP BY c_a.element, c_b.element
ORDER BY stable_percentage DESC, avg_energy_above_hull ASC, l12_combination_count DESC
LIMIT 10000;
