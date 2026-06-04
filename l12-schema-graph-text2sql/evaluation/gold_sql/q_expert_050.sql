SELECT m.entry_id, m.formula, mp.total_magnetization, mp.magnetic_ordering
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.total_magnetization != 0 AND mp.total_magnetization IS NOT NULL
ORDER BY mp.total_magnetization DESC
LIMIT 10000;
