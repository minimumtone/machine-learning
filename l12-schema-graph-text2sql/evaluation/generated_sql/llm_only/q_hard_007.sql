SELECT
  fh.entry_id,
  fh.formula,
  fh.reduced_formula,
  fh.formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
  fh.lattice_a,
  fh.prototype,
  fh.strukturbericht
FROM formation_enthalpy AS fh
WHERE fh.formation_enthalpy_ev_per_atom <= -0.3
  AND fh.lattice_a BETWEEN 3.5 AND 3.7
  AND (
    fh.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR fh.prototype ILIKE '%L1_2%'
    OR fh.prototype ILIKE '%L12%'
    OR fh.prototype ILIKE '%L1₂%'
  );
