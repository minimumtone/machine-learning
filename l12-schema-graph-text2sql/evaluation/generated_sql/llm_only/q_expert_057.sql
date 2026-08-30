SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    tp.thermal_conductivity,
    tp.temperature_k
FROM material_entry AS me
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN thermal_property AS tp
    ON tp.calculation_id = c.calculation_id
WHERE tp.thermal_conductivity IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM structure AS s
      WHERE s.entry_id = me.entry_id
        AND s.strukturbericht = 'L12'
  )
ORDER BY me.reduced_formula, me.entry_id, tp.temperature_k;
