SELECT
  c_a.element AS a_site,
  c_b.element AS b_site,
  COUNT(*) AS count,
  COUNT(*) FILTER (WHERE ps.is_stable = TRUE) AS count,
  COUNT(*) FILTER (WHERE ps.is_stable = FALSE) AS unstable_count,
  COUNT(*) FILTER (WHERE ps.is_stable = TRUE) * 100.0 / COUNT(*) AS stable_percentage,
  AVG(ps.energy_above_hull) AS avg_energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c_a ON c_a.entry_id = m.entry_id
JOIN composition c_b ON c_b.entry_id = m.entry_id
WHERE s.prototype = 'BiF3'
  AND c_a.site_label = 'A-site'
  AND c_b.site_label = 'B-site'
GROUP BY c_a.element, c_b.element
ORDER BY stable_percentage DESC, avg_energy_above_hull ASC
LIMIT 10000;
