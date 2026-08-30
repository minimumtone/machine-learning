SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN magnetic_property AS mp
  ON mp.entry_id = me.entry_id
WHERE lower(mp.magnetic_ordering) = 'ferromagnetic'
  AND (
      replace(upper(COALESCE(s.strukturbericht, '')), '_', '') = 'L12'
      OR replace(upper(COALESCE(s.prototype, '')), '_', '') = 'L12'
  )
  AND EXISTS (
      SELECT 1
      FROM calculation AS c
      JOIN elastic_tensor AS et
        ON et.calculation_id = c.calculation_id
      WHERE c.entry_id = me.entry_id
        AND et.is_stable = TRUE
  );
