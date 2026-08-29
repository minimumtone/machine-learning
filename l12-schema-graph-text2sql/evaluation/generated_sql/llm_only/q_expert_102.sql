WITH elastic AS (
  SELECT c.entry_id, MAX(et.bulk_modulus_vrh) AS bulk_modulus_gpa
  FROM calculation c
  JOIN elastic_tensor et ON et.calculation_id = c.calculation_id
  WHERE et.bulk_modulus_vrh >= 150
  GROUP BY c.entry_id
),
thermal AS (
  SELECT c.entry_id, MAX(tp.debye_temperature_k) AS debye_temperature_k
  FROM calculation c
  JOIN thermal_property tp ON tp.calculation_id = c.calculation_id
  WHERE tp.debye_temperature_k >= 400
  GROUP BY c.entry_id
),
stable AS (
  SELECT entry_id, MIN(energy_above_hull) AS energy_above_hull
  FROM phase_stability
  WHERE is_stable = TRUE
  GROUP BY entry_id
)
SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  s.strukturbericht,
  e.bulk_modulus_gpa,
  t.debye_temperature_k,
  st.energy_above_hull
FROM material_entry me
JOIN composition co ON co.entry_id = me.entry_id
JOIN structure s ON s.entry_id = me.entry_id
JOIN stable st ON st.entry_id = me.entry_id
JOIN elastic e ON e.entry_id = me.entry_id
JOIN thermal t ON t.entry_id = me.entry_id
WHERE co.element = 'Ni'
  AND s.strukturbericht IN ('L12', 'L1_2', 'L1₂')
ORDER BY e.bulk_modulus_gpa DESC, t.debye_temperature_k DESC;
