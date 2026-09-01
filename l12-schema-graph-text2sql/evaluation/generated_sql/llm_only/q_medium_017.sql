SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.prototype,
  s.strukturbericht,
  s.space_group,
  ps.reference_set,
  ps.formation_energy_per_atom,
  ps.energy_above_hull,
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
    ELSE 'unstable'
  END AS stability_class,
  ps.is_stable,
  ps.band_gap
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
WHERE c.element = 'Nb'
  AND s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
ORDER BY ps.energy_above_hull ASC NULLS LAST, me.entry_id;
