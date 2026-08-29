SELECT c_a.element AS a_site,
       c_b.element AS b_site,
       COUNT(*) AS nias_compound_count,
       AVG(ps.energy_above_hull) AS avg_energy_above_hull,
       SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) AS count,
       SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS stable_percentage
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c_a ON c_a.entry_id = m.entry_id
JOIN composition c_b ON c_b.entry_id = m.entry_id
WHERE (s.prototype = 'NiAs' OR s.strukturbericht = 'NiAs')
  AND c_a.site_label = 'A-site'
  AND c_b.site_label = 'B-site'
GROUP BY c_a.element, c_b.element
ORDER BY stable_percentage DESC, avg_energy_above_hull ASC, nias_compound_count DESC
LIMIT 10000;
