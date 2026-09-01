SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht,
    ps.energy_above_hull
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE ps.energy_above_hull > 0.001
  AND ps.energy_above_hull <= 0.05
  AND (
      UPPER(REPLACE(COALESCE(s.strukturbericht, ''), '_', '')) = 'L12'
      OR UPPER(REPLACE(COALESCE(pd.strukturbericht, ''), '_', '')) = 'L12'
      OR UPPER(REPLACE(COALESCE(s.prototype, ''), '_', '')) LIKE '%L12%'
      OR UPPER(REPLACE(COALESCE(pd.prototype_name, ''), '_', '')) LIKE '%L12%'
  )
ORDER BY ps.energy_above_hull, me.reduced_formula;
