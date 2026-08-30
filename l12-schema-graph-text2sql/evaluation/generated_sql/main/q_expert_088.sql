SELECT COUNT(*) AS stable_binary_compound_count
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.number_of_elements = 2
  AND ps.is_stable = TRUE
LIMIT 10000;
