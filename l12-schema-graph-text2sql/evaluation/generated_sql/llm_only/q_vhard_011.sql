SELECT
  DENSE_RANK() OVER (ORDER BY ps.energy_above_hull ASC) AS rank,
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ps.reference_set,
  ps.energy_above_hull,
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
    ELSE 'unstable'
  END AS stability_class
FROM material_entry me
JOIN phase_stability ps
  ON ps.entry_id = me.entry_id
JOIN structure s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition pd
  ON pd.prototype_id = s.prototype
WHERE COALESCE(s.strukturbericht, pd.strukturbericht) IN ('L1_2', 'L12', 'L1₂')
ORDER BY ps.energy_above_hull ASC, me.entry_id;
