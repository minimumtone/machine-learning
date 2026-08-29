SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    s.space_group_number,
    s.space_group,
    s.crystal_system
FROM material_entry AS me
JOIN structure AS s
    ON me.entry_id = s.entry_id
WHERE LOWER(s.crystal_system) = 'cubic'
  AND me.number_of_elements > 1
ORDER BY me.reduced_formula, me.entry_id;
