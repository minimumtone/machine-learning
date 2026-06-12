SELECT ca.element AS a_site, cb.element AS b_site,
       AVG(ps.formation_energy_per_atom) AS avg_eform,
       AVG(s.lattice_a) AS avg_lattice,
       AVG(cp_bm.value) AS avg_bulk_modulus,
       SUM(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1 ELSE 0 END) AS stable_count
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A-site'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B-site'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
GROUP BY ca.element, cb.element
ORDER BY avg_eform ASC
LIMIT 10000;