SELECT DISTINCT
  me.formula,
  s.prototype
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE s.strukturbericht = 'L12'
ORDER BY me.formula, s.prototype;
