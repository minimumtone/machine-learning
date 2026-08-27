SELECT
  CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable' ELSE 'unstable' END AS stability,
  AVG(et.bulk_modulus_vrh) AS avg_bulk,
  COUNT(*) AS cnt
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
JOIN calculation cal_et ON cal_et.entry_id = s.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
GROUP BY ps.energy_above_hull <= 0.001
LIMIT 10000;
