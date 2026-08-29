SELECT ca.element AS a_site,
       cb.element AS b_site,
       AVG(cp.value) AS avg_bulk_modulus,
       COUNT(DISTINCT m.entry_id) FILTER (WHERE ps.is_stable = TRUE) AS stable_phase_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition ca ON ca.entry_id = m.entry_id
JOIN composition cb ON cb.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ca.site_label = 'A-site'
  AND cb.site_label = 'B-site'
  AND cp.property_name = 'bulk_modulus'
GROUP BY ca.element, cb.element
ORDER BY stable_phase_count DESC, avg_bulk_modulus DESC
LIMIT 10000;
