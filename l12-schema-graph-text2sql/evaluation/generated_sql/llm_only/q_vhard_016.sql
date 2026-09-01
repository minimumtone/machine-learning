WITH candidates AS (
  SELECT
    fh.entry_id,
    fh.formula,
    fh.reduced_formula,
    fh.reference_set,
    COALESCE(fh.strukturbericht, s.strukturbericht) AS strukturbericht,
    COALESCE(fh.prototype, s.prototype) AS prototype,
    COALESCE(
      fh.enthalpy_vs_element_ground_states,
      fh.formation_enthalpy_ev_per_atom + fh.weighted_element_delta_e
    ) AS formation_energy_vs_element_ground_states_ev_per_atom,
    COALESCE(s.lattice_a, fh.lattice_a) AS lattice_a,
    COALESCE(s.lattice_b, s.lattice_a, fh.lattice_a) AS lattice_b,
    COALESCE(s.lattice_c, s.lattice_a, fh.lattice_a) AS lattice_c,
    COALESCE(s.space_group, fh.space_group) AS space_group
  FROM formation_enthalpy fh
  LEFT JOIN structure s
    ON s.entry_id = fh.entry_id
  WHERE fh.is_stable = TRUE
    AND (
      regexp_replace(lower(replace(COALESCE(fh.strukturbericht, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
      OR regexp_replace(lower(replace(COALESCE(s.strukturbericht, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
      OR regexp_replace(lower(replace(COALESCE(fh.prototype, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
      OR regexp_replace(lower(replace(COALESCE(s.prototype, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
      OR COALESCE(fh.prototype, '') ILIKE '%Cu3Au%'
      OR COALESCE(s.prototype, '') ILIKE '%Cu3Au%'
    )
)
SELECT DISTINCT
  entry_id,
  formula,
  reduced_formula,
  reference_set,
  strukturbericht,
  prototype,
  formation_energy_vs_element_ground_states_ev_per_atom,
  lattice_a,
  lattice_b,
  lattice_c,
  space_group
FROM candidates
WHERE formation_energy_vs_element_ground_states_ev_per_atom < -0.3
ORDER BY formation_energy_vs_element_ground_states_ev_per_atom;
