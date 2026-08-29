SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    c.atomic_fraction AS pt_atomic_fraction,
    s.prototype,
    s.strukturbericht
FROM material_entry AS me
JOIN composition AS c
    ON c.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE c.element = 'Pt'
  AND c.atomic_fraction >= 0.25
  AND (
      UPPER(REPLACE(COALESCE(s.strukturbericht, ''), '_', '')) = 'L12'
      OR UPPER(REPLACE(COALESCE(pd.strukturbericht, ''), '_', '')) = 'L12'
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1_2%'
      OR pd.prototype_name ILIKE '%L12%'
      OR pd.prototype_name ILIKE '%L1_2%'
  );
