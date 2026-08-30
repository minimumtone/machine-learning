SELECT
  entry_id,
  formula,
  reduced_formula,
  prototype,
  strukturbericht,
  lattice_a,
  formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
  energy_above_hull
FROM formation_enthalpy
WHERE replace(translate(upper(strukturbericht), '₂', '2'), '_', '') = 'L12'
  AND lattice_a IS NOT NULL
  AND abs(lattice_a - 3.57) <= 0.1
ORDER BY formation_enthalpy_ev_per_atom ASC;
