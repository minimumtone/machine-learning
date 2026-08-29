SELECT m.formula, ps.energy_above_hull, ps.is_stable, mp.magnetic_ordering
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.magnetic_ordering = 'ferromagnetic'
  AND ps.is_stable = TRUE
ORDER BY m.formula ASC
LIMIT 10000;
