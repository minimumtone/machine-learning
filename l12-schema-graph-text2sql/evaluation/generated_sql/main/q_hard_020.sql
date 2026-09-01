SELECT c.element AS b_site,
       COUNT(DISTINCT m.entry_id) AS stable_l12_count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'B-site'
  AND ps.is_stable = TRUE
GROUP BY c.element
ORDER BY stable_l12_count DESC
LIMIT 10000;
