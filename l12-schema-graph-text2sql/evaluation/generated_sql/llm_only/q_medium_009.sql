SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.strukturbericht,
    s.lattice_a
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE s.lattice_a >= 3.5
  AND UPPER(REPLACE(REPLACE(s.strukturbericht, '_', ''), '₂', '2')) = 'L12'
ORDER BY s.lattice_a, me.formula;
