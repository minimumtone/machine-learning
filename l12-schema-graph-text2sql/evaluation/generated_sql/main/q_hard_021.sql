SELECT
    m.formula,
    m.source_db,
    s.prototype,
    s.strukturbericht,
    s.crystal_system,
    s.space_group_number,
    s.space_group,
    s.lattice_a,
    s.lattice_b,
    s.lattice_c,
    s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12'
   OR s.strukturbericht = 'L12'
ORDER BY m.formula
LIMIT 10000;
