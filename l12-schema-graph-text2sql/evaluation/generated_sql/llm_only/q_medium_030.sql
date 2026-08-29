SELECT
  entry_id,
  formula,
  reduced_formula,
  lattice_a
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND strukturbericht IN ('L1_2', 'L12', 'L1₂')
  AND lattice_a IS NOT NULL
ORDER BY lattice_a DESC;
