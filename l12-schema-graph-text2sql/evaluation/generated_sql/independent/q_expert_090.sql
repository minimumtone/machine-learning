SELECT m.formula, se.miller_index, se.surface_energy_j_m2, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND se.surface_energy_j_m2 < 1.5
  AND ps.is_stable = TRUE
ORDER BY se.surface_energy_j_m2 ASC
LIMIT 10000;
