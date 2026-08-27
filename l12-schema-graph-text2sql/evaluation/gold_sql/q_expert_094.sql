SELECT m.entry_id, m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a IS NOT NULL
  AND ps.formation_energy_per_atom IS NOT NULL
ORDER BY s.lattice_a, m.entry_id ASC
LIMIT 10000;
