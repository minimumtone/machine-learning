SELECT m.formula,
       ps.is_stable,
       ps.energy_above_hull,
       s.lattice_a,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.element = 'Co'
  AND ps.is_stable = TRUE
ORDER BY ABS(s.lattice_a - 3.57) ASC, ps.energy_above_hull ASC
LIMIT 10000;
