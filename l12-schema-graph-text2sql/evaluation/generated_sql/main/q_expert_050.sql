SELECT m.formula, s.prototype, s.strukturbericht, mp.total_magnetization
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.total_magnetization <> 0
ORDER BY ABS(mp.total_magnetization) DESC
LIMIT 10000;
