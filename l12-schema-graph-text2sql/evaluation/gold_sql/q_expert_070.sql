SELECT m.entry_id, m.formula, s.prototype, s.lattice_a, s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype IN ('L12', 'B2') OR s.strukturbericht IN ('L12', 'B2')
ORDER BY s.volume_per_atom DESC, m.entry_id ASC
LIMIT 10000;
