SELECT
  entry_id,
  formula,
  reduced_formula,
  lattice_a,
  energy_above_hull,
  formation_enthalpy_ev_per_atom
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND lattice_a < 3.6
  AND strukturbericht IN ('L1_2', 'L12', 'L1₂')
ORDER BY lattice_a ASC, formula;
