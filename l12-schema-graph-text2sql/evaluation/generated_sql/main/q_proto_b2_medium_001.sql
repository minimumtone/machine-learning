SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group
FROM material_entry m
    JOIN composition c ON c.entry_id = m.entry_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'B2' OR s.strukturbericht = 'B2')
    AND c.element = 'Zr'

LIMIT 10000;
