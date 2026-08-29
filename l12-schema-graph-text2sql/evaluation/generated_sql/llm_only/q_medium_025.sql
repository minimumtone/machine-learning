SELECT
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
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE EXISTS (
  SELECT 1
  FROM composition AS c
  WHERE c.entry_id = me.entry_id
    AND c.element = 'Cu'
)
AND (
  REPLACE(REPLACE(UPPER(COALESCE(s.strukturbericht, pd.strukturbericht, '')), '_', ''), '₂', '2') = 'L12'
  OR s.prototype ILIKE '%Cu3Au%'
  OR pd.prototype_name ILIKE '%Cu3Au%'
)
ORDER BY
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 1
    WHEN ps.energy_above_hull <= 0.05 THEN 2
    ELSE 3
  END,
  ps.energy_above_hull,
  me.reduced_formula;
