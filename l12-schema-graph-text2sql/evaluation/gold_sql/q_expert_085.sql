SELECT m.entry_id, m.formula, mp.magnetic_ordering, et.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.magnetic_ordering = 'ferromagnetic'
  AND et.is_stable = TRUE
ORDER BY m.formula
LIMIT 10000;
