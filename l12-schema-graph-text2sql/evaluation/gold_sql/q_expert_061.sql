SELECT m.entry_id, m.formula, se.surface_energy_j_m2
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND se.miller_index = '111'
ORDER BY se.surface_energy_j_m2 ASC
LIMIT 1;
