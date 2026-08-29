SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht,
    s.lattice_a
FROM structure AS s
JOIN material_entry AS me
    ON me.entry_id = s.entry_id
WHERE s.lattice_a BETWEEN 3.50 AND 3.60
  AND UPPER(REGEXP_REPLACE(COALESCE(s.strukturbericht, ''), '[^A-Za-z0-9]', '', 'g')) = 'L12'
ORDER BY s.lattice_a, me.formula;
