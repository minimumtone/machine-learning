SELECT AVG(e.energy_above_hull) AS average_energy_above_hull
FROM mp_entries e
JOIN mp_element_ratios r1 ON e.entry_id = r1.entry_id
JOIN mp_element_ratios r2 ON e.entry_id = r2.entry_id
WHERE r1.element = 'Co'
  AND r2.element = 'Ti'
LIMIT 10000;
