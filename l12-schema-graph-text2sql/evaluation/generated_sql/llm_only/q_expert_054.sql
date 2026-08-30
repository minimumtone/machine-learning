SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    mp.total_magnetization,
    mp.curie_temperature_k
FROM material_entry AS me
JOIN magnetic_property AS mp
    ON mp.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
WHERE mp.magnetic_ordering = 'ferromagnetic'
  AND UPPER(REGEXP_REPLACE(s.strukturbericht, '[^A-Za-z0-9]', '', 'g')) = 'L12'
ORDER BY me.reduced_formula, me.entry_id;
