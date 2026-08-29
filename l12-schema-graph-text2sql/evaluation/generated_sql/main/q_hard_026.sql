SELECT m.formula,
       ps.energy_above_hull,
       ps.is_stable,
       s.lattice_a,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY ps.energy_above_hull ASC,
         ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;
