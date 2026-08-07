SELECT ca.element AS a_site, cb.element AS b_site,
       COUNT(*) AS total,
       SUM(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1 ELSE 0 END) AS stable_count,
       AVG(ps.formation_energy_per_atom) AS avg_eform
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A-site'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B-site'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'BiF3' OR s.strukturbericht = 'D0_3'
GROUP BY ca.element, cb.element
ORDER BY avg_eform ASC
LIMIT 10000;