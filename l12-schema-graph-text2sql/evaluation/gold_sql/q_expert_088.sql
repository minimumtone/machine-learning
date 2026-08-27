SELECT s.prototype, COUNT(*) AS cnt
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.number_of_elements = 2
  AND ps.energy_above_hull <= 0.001
GROUP BY s.prototype
ORDER BY cnt DESC, s.prototype ASC
LIMIT 10000;
