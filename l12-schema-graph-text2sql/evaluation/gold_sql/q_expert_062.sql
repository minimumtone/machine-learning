SELECT DISTINCT m.entry_id, m.formula, se.work_function, se.miller_index
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND se.work_function >= 5.0
ORDER BY se.work_function DESC
LIMIT 10000;
