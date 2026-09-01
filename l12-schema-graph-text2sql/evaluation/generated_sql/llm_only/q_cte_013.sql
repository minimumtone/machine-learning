SELECT
  me.entry_id,
  me.formula,
  fe.reduced_formula,
  ps.reference_set,
  ps.formation_energy_per_atom AS formation_energy_ev_per_atom,
  fe.enthalpy_vs_element_ground_states AS pure_element_corrected_formation_enthalpy_ev_per_atom,
  ABS(ps.formation_energy_per_atom - fe.enthalpy_vs_element_ground_states) AS absolute_difference_ev_per_atom,
  COALESCE(s.strukturbericht, fe.strukturbericht) AS strukturbericht,
  COALESCE(s.prototype, fe.prototype) AS prototype
FROM material_entry AS me
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN formation_enthalpy AS fe
  ON fe.entry_id = me.entry_id
 AND fe.reference_set = ps.reference_set
LEFT JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE ps.formation_energy_per_atom IS NOT NULL
  AND fe.enthalpy_vs_element_ground_states IS NOT NULL
  AND (
    COALESCE(s.strukturbericht, fe.strukturbericht) ILIKE 'L12'
    OR COALESCE(s.strukturbericht, fe.strukturbericht) ILIKE 'L1_2'
    OR COALESCE(s.strukturbericht, fe.strukturbericht) ILIKE 'L1₂'
    OR COALESCE(s.prototype, fe.prototype) ILIKE '%L12%'
    OR COALESCE(s.prototype, fe.prototype) ILIKE '%L1_2%'
    OR COALESCE(s.prototype, fe.prototype) ILIKE '%L1₂%'
  )
ORDER BY absolute_difference_ev_per_atom DESC
LIMIT 10;
