SELECT m.entry_id, m.formula, s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.volume_per_atom IS NOT NULL
ORDER BY s.volume_per_atom ASC
LIMIT 1;
