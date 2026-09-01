SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.lattice_a ASC
LIMIT 10000;
