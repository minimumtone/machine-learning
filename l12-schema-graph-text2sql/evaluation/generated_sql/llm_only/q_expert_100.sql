SELECT
  pd.prototype_id,
  pd.prototype_name,
  COUNT(ps.stability_id) FILTER (WHERE ps.energy_above_hull <= 0.001) AS stable_count,
  COUNT(ps.stability_id) FILTER (WHERE ps.energy_above_hull > 0.05) AS unstable_count,
  COUNT(ps.stability_id) FILTER (WHERE ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05) AS metastable_count
FROM prototype_definition pd
LEFT JOIN structure s
  ON s.prototype = pd.prototype_id
LEFT JOIN phase_stability ps
  ON ps.entry_id = s.entry_id
GROUP BY
  pd.prototype_id,
  pd.prototype_name
ORDER BY
  pd.prototype_id;
