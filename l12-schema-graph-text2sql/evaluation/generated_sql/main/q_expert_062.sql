SELECT m.formula, se.miller_index, se.work_function
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND se.work_function >= 5
ORDER BY se.work_function DESC
LIMIT 10000;
