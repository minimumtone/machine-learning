SELECT ca.element AS a_site, cb.element AS b_site,
       m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A-site'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B-site'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY cp_bm.value DESC
LIMIT 10000;