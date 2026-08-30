SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group
FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND s.lattice_a >= 4.0

LIMIT 10000;
