SELECT m.formula, s.lattice_a, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY s.lattice_a DESC
LIMIT 10000;