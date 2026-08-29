SELECT c_a.element AS a_site,
       c_b.element AS b_site,
       COUNT(*) AS nacl_compound_count,
       AVG(ps.energy_above_hull) AS avg_energy_above_hull,
       SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) AS count,
       SUM(CASE WHEN ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05 THEN 1 ELSE 0 END) AS metastable_count,
       SUM(CASE WHEN ps.energy_above_hull > 0.05 THEN 1 ELSE 0 END) AS unstable_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c_a ON c_a.entry_id = m.entry_id
JOIN composition c_b ON c_b.entry_id = m.entry_id
WHERE (s.prototype = 'NaCl' OR s.strukturbericht = 'B1')
  AND c_a.site_label = 'A-site'
  AND c_b.site_label = 'B-site'
GROUP BY c_a.element, c_b.element
ORDER BY avg_energy_above_hull ASC
LIMIT 10000;
