SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    mp.total_magnetization,
    mp.magnetic_ordering
FROM material_entry AS me
JOIN magnetic_property AS mp
    ON mp.entry_id = me.entry_id
WHERE mp.total_magnetization IS NOT NULL
  AND mp.total_magnetization <> 0
  AND (
      EXISTS (
          SELECT 1
          FROM structure AS s
          WHERE s.entry_id = me.entry_id
            AND (
                UPPER(REPLACE(REPLACE(COALESCE(s.strukturbericht, ''), '_', ''), '-', '')) = 'L12'
                OR UPPER(REPLACE(REPLACE(COALESCE(s.prototype, ''), '_', ''), '-', '')) = 'L12'
            )
      )
      OR EXISTS (
          SELECT 1
          FROM formation_enthalpy AS fe
          WHERE fe.entry_id = me.entry_id
            AND (
                UPPER(REPLACE(REPLACE(COALESCE(fe.strukturbericht, ''), '_', ''), '-', '')) = 'L12'
                OR UPPER(REPLACE(REPLACE(COALESCE(fe.prototype, ''), '_', ''), '-', '')) = 'L12'
            )
      )
  );
