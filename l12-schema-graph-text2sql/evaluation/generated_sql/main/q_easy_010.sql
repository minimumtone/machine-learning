SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group
FROM material_entry m
    JOIN composition c ON c.entry_id = m.entry_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND c.element = 'Ti'

LIMIT 10000;
