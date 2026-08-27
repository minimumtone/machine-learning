SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, ps.energy_above_hull
FROM material_entry m
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE et.bulk_modulus_vrh > (
    SELECT et2.bulk_modulus_vrh FROM elastic_tensor et2
    JOIN calculation cal_et2 ON cal_et2.calculation_id = et2.calculation_id
    JOIN material_entry m2 ON m2.entry_id = cal_et2.entry_id
    WHERE m2.formula = 'Ni3Al' LIMIT 1
  )
  AND ps.energy_above_hull <= 0.01
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;
