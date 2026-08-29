SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  c.atomic_fraction AS ni_atomic_fraction,
  s.prototype,
  s.strukturbericht,
  s.space_group_number,
  s.space_group,
  ps.reference_set,
  ps.formation_energy_per_atom,
  ps.energy_above_hull,
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
  END AS stability_class,
  ps.band_gap
FROM material_entry me
JOIN composition c
  ON c.entry_id = me.entry_id
 AND c.element = 'Ni'
JOIN structure s
  ON s.entry_id = me.entry_id
JOIN phase_stability ps
  ON ps.entry_id = me.entry_id
WHERE c.atomic_fraction >= 0.5
  AND ps.energy_above_hull <= 0.05
  AND (
    s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR s.prototype ILIKE '%L1_2%'
    OR s.prototype ILIKE '%L12%'
    OR s.prototype ILIKE '%Cu3Au%'
  )
ORDER BY ps.energy_above_hull ASC, c.atomic_fraction DESC, me.reduced_formula;
