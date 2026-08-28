SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ABS(s.lattice_a - 3.57) <= 0.1
ORDER BY ps.formation_energy_per_atom ASC, m.entry_id ASC
LIMIT 10000;