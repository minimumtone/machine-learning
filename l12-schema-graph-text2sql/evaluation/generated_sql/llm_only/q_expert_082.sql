SELECT
  RANK() OVER (ORDER BY formation_enthalpy_ev_per_atom ASC) AS formation_energy_rank,
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
  energy_above_hull,
  CASE
    WHEN energy_above_hull <= 0.001 THEN 'stable'
    ELSE 'metastable'
  END AS stability_class,
  prototype,
  strukturbericht,
  space_group,
  reference_set
FROM formation_enthalpy
WHERE energy_above_hull <= 0.05
  AND (
    regexp_replace(lower(coalesce(strukturbericht, '')), '[^a-z0-9]', '', 'g') = 'l12'
    OR regexp_replace(lower(coalesce(prototype, '')), '[^a-z0-9]', '', 'g') LIKE '%l12%'
    OR regexp_replace(lower(coalesce(prototype, '')), '[^a-z0-9]', '', 'g') LIKE '%cu3au%'
  )
ORDER BY formation_enthalpy_ev_per_atom ASC, energy_above_hull ASC, entry_id;
