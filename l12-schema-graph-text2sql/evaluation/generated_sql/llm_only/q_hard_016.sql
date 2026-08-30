SELECT
  entry_id,
  formula,
  reduced_formula,
  lattice_a,
  formation_enthalpy_ev_per_atom,
  corr(lattice_a, formation_enthalpy_ev_per_atom) OVER () AS pearson_corr_lattice_a_formation_energy
FROM formation_enthalpy
WHERE
  (
    upper(replace(strukturbericht, '_', '')) IN ('L12', 'L1₂')
    OR upper(replace(prototype, '_', '')) IN ('L12', 'L1₂')
  )
  AND lattice_a IS NOT NULL
  AND formation_enthalpy_ev_per_atom IS NOT NULL
ORDER BY lattice_a;
