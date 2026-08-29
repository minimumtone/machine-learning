SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  tp.debye_temperature_k
FROM thermal_property tp
JOIN calculation c
  ON tp.calculation_id = c.calculation_id
JOIN material_entry me
  ON c.entry_id = me.entry_id
WHERE tp.debye_temperature_k >= 500
  AND me.number_of_elements > 1
ORDER BY tp.debye_temperature_k DESC;
