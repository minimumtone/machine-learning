SELECT m.formula, s.lattice_a,
       ABS(s.lattice_a - 3.57) AS mismatch
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY ABS(s.lattice_a - 3.57) ASC
LIMIT 100;