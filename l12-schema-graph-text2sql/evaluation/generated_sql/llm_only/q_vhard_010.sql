SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.prototype,
  s.strukturbericht,
  ps.reference_set,
  ps.formation_energy_per_atom AS formation_energy_ev_per_atom,
  ps.energy_above_hull AS energy_above_hull_ev_per_atom,
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
  END AS stability_class,
  et.bulk_modulus_vrh,
  et.shear_modulus_vrh,
  et.youngs_modulus,
  et.poisson_ratio
FROM material_entry me
JOIN structure s
  ON s.entry_id = me.entry_id
JOIN phase_stability ps
  ON ps.entry_id = me.entry_id
LEFT JOIN calculation c
  ON c.entry_id = me.entry_id
LEFT JOIN elastic_tensor et
  ON et.calculation_id = c.calculation_id
WHERE ps.energy_above_hull <= 0.05
  AND (
    s.strukturbericht = 'L1_2'
    OR s.prototype ILIKE '%L1_2%'
    OR s.prototype ILIKE '%L12%'
  )
ORDER BY
  ps.energy_above_hull ASC,
  ps.formation_energy_per_atom ASC;
