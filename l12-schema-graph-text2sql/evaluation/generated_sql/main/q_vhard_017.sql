SELECT
  m.formula,
  ps.energy_above_hull,
  s.lattice_a,
  ABS(s.lattice_a - 3.57) AS lattice_diff,
  ps.formation_energy_per_atom,
  (
    (1 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.30
    + (1 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.30
    + 0.0 * 0.20
    + (-ps.formation_energy_per_atom / 1.0) * 0.20
  ) AS composite_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY composite_score DESC
LIMIT 20;
