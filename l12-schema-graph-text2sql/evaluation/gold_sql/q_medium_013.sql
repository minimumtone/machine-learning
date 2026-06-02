SELECT DISTINCT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Ti'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;