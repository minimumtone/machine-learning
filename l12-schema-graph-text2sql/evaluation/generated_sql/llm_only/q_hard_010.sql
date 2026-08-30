SELECT
  me.entry_id,
  me.formula,
  c.element AS b_site_element,
  ps.formation_energy_per_atom,
  ps.energy_above_hull,
  ps.is_stable
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN composition AS c
  ON c.entry_id = me.entry_id
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
WHERE c.site_label = 'B-site'
  AND c.element IN ('Al', 'Ti')
  AND s.strukturbericht IN ('L1_2', 'L1₂')
ORDER BY ps.formation_energy_per_atom ASC;
