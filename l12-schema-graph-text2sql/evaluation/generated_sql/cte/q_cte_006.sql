SELECT DISTINCT ON (c.element)
       c.element AS a_site,
       m.formula,
       ps.formation_energy_per_atom AS rebaselined_formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A-site'
ORDER BY c.element, ps.formation_energy_per_atom ASC
LIMIT 10000;
