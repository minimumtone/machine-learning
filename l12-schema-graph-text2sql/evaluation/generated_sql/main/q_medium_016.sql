SELECT m.formula, ps.energy_above_hull, ps.is_stable, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND s.lattice_a < 3.6
ORDER BY s.lattice_a ASC
LIMIT 10000;
