SELECT e.formula, e.energy_above_hull
FROM mp_entries e
JOIN mp_element_ratios r ON e.entry_id = r.entry_id
WHERE r.element = 'Co'
ORDER BY e.energy_above_hull ASC
LIMIT 1;
