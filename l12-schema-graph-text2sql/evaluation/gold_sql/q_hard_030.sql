SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND c.element IN ('Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                     'Zr','Nb','Mo','Ru','Rh','Pd','Ag',
                     'Hf','Ta','W','Re','Os','Ir','Pt','Au')
ORDER BY ps.energy_above_hull ASC
LIMIT 100;