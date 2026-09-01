SELECT m.formula,
       ca.element AS a_site,
       ca.atomic_fraction AS a_site_atomic_fraction,
       cb.element AS b_site,
       cb.atomic_fraction AS b_site_atomic_fraction,
       ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition ca ON ca.entry_id = m.entry_id
JOIN composition cb ON cb.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ca.site_label = 'A-site'
  AND cb.site_label = 'B-site'
ORDER BY ca.element, cb.element, ps.formation_energy_per_atom
LIMIT 10000;
