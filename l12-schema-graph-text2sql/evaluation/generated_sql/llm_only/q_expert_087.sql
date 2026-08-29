WITH ni3al_bulk AS (
  SELECT MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh
  FROM material_entry me
  JOIN calculation c ON c.entry_id = me.entry_id
  JOIN elastic_tensor et ON et.calculation_id = c.calculation_id
  WHERE me.formula IN ('Ni3Al', 'AlNi3')
     OR me.reduced_formula IN ('Ni3Al', 'AlNi3')
)
SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  et.bulk_modulus_vrh,
  ps.energy_above_hull
FROM material_entry me
JOIN phase_stability ps ON ps.entry_id = me.entry_id
JOIN calculation c ON c.entry_id = me.entry_id
JOIN elastic_tensor et ON et.calculation_id = c.calculation_id
CROSS JOIN ni3al_bulk nb
WHERE ps.energy_above_hull <= 0.01
  AND et.bulk_modulus_vrh > nb.bulk_modulus_vrh
ORDER BY et.bulk_modulus_vrh DESC;
