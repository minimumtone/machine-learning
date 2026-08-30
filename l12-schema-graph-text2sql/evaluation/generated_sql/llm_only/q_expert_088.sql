SELECT
  s.prototype,
  COUNT(DISTINCT me.entry_id) AS stable_binary_compound_count
FROM material_entry AS me
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE me.number_of_elements = 2
  AND ps.is_stable = TRUE
GROUP BY s.prototype
ORDER BY stable_binary_compound_count DESC, s.prototype;
