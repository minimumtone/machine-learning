SELECT
  ca.element AS a_site,
  cb.element AS b_site,
  COUNT(*) AS count,
  COUNT(*) FILTER (WHERE ps.is_stable = TRUE) AS count,
  COUNT(*) FILTER (WHERE ps.is_stable = FALSE) AS unstable_count,
  COUNT(*) FILTER (WHERE ps.is_stable = TRUE) * 100.0 / COUNT(*) AS stable_percentage,
  AVG(ps.energy_above_hull) AS avg_energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition ca ON ca.entry_id = m.entry_id
JOIN composition cb ON cb.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND ca.site_label = 'A-site'
  AND cb.site_label = 'B-site'
GROUP BY ca.element, cb.element
ORDER BY stable_percentage DESC, avg_energy_above_hull ASC
LIMIT 10000;
