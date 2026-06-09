SELECT m.entry_id, m.formula, mp.total_magnetization, mp.curie_temperature_k
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.magnetic_ordering = 'ferromagnetic'
ORDER BY mp.total_magnetization DESC
LIMIT 10000;
