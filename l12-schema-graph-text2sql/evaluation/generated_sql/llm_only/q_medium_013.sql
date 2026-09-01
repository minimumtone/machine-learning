SELECT
  me.entry_id,
  me.formula,
  s.lattice_a,
  s.lattice_b,
  s.lattice_c,
  ps.formation_energy_per_atom
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
WHERE c.element = 'Ti'
  AND s.strukturbericht = 'L1_2'
ORDER BY me.formula;
