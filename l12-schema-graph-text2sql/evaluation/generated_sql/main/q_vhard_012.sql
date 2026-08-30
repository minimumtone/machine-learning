SELECT
  m.formula,
  ps.is_stable,
  ps.energy_above_hull,
  ps.formation_energy_per_atom,
  s.lattice_a,
  ABS(s.lattice_a - 3.57) AS lattice_diff,
  (
    (1 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.30
    + (1 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.25
    + CASE
        WHEN ps.formation_energy_per_atom < -0.3 THEN 0.20
        ELSE 0.10
      END
  ) AS design_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.chemical_system != 'Al-Ni'
  AND ps.is_stable = TRUE
ORDER BY design_score DESC
LIMIT 10000;
