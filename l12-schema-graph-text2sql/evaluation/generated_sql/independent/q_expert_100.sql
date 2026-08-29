SELECT
  COUNT(*) FILTER (WHERE ps.energy_above_hull <= 0.001) AS count,
  COUNT(*) FILTER (WHERE ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05) AS metastable_count,
  COUNT(*) FILTER (WHERE ps.energy_above_hull > 0.05) AS unstable_count
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
LIMIT 10000;
