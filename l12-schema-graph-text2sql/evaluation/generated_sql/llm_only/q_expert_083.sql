SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  MAX(et.bulk_modulus_vrh) AS bulk_modulus_gpa,
  MAX(tp.debye_temperature_k) AS debye_temperature_k
FROM material_entry AS me
JOIN composition AS comp
  ON comp.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
JOIN thermal_property AS tp
  ON tp.calculation_id = c.calculation_id
WHERE comp.element = 'Co'
  AND ps.is_stable = TRUE
  AND UPPER(REPLACE(s.strukturbericht, '_', '')) = 'L12'
  AND et.bulk_modulus_vrh >= 180
  AND tp.debye_temperature_k >= 400
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula;
