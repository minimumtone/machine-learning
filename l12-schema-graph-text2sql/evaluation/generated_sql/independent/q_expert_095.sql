SELECT
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05 THEN 'metastable'
    WHEN ps.energy_above_hull > 0.05 THEN 'unstable'
  END AS stability,
  AVG(cp.value) AS avg_bulk_modulus,
  COUNT(*) AS count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
GROUP BY
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05 THEN 'metastable'
    WHEN ps.energy_above_hull > 0.05 THEN 'unstable'
  END
ORDER BY
  CASE
    WHEN CASE
      WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
      WHEN ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05 THEN 'metastable'
      WHEN ps.energy_above_hull > 0.05 THEN 'unstable'
    END = 'stable' THEN 1
    WHEN CASE
      WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
      WHEN ps.energy_above_hull > 0.001 AND ps.energy_above_hull <= 0.05 THEN 'metastable'
      WHEN ps.energy_above_hull > 0.05 THEN 'unstable'
    END = 'metastable' THEN 2
    ELSE 3
  END
LIMIT 10000;
