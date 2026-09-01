WITH candidates AS (
  SELECT DISTINCT
    me.entry_id,
    me.formula,
    f.reduced_formula,
    f.reference_set,
    f.formation_enthalpy_ev_per_atom,
    f.weighted_element_delta_e,
    COALESCE(
      f.enthalpy_vs_element_ground_states,
      f.formation_enthalpy_ev_per_atom + f.weighted_element_delta_e
    ) AS rereferenced_formation_energy_ev_per_atom,
    mp.total_magnetization,
    mp.magnetic_ordering
  FROM formation_enthalpy f
  JOIN material_entry me
    ON me.entry_id = f.entry_id
  JOIN magnetic_property mp
    ON mp.entry_id = f.entry_id
  WHERE
    (
      LOWER(mp.magnetic_ordering) LIKE 'ferro%'
      OR UPPER(mp.magnetic_ordering) = 'FM'
    )
    AND (
      regexp_replace(
        translate(UPPER(COALESCE(f.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'),
        '[^A-Z0-9]', '', 'g'
      ) = 'L12'
      OR regexp_replace(
        translate(UPPER(COALESCE(f.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'),
        '[^A-Z0-9]', '', 'g'
      ) LIKE '%L12%'
      OR EXISTS (
        SELECT 1
        FROM structure s
        WHERE s.entry_id = f.entry_id
          AND (
            regexp_replace(
              translate(UPPER(COALESCE(s.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'),
              '[^A-Z0-9]', '', 'g'
            ) = 'L12'
            OR regexp_replace(
              translate(UPPER(COALESCE(s.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'),
              '[^A-Z0-9]', '', 'g'
            ) LIKE '%L12%'
          )
      )
    )
)
SELECT
  entry_id,
  formula,
  reduced_formula,
  reference_set,
  formation_enthalpy_ev_per_atom,
  weighted_element_delta_e,
  rereferenced_formation_energy_ev_per_atom,
  total_magnetization,
  magnetic_ordering
FROM candidates
ORDER BY rereferenced_formation_energy_ev_per_atom ASC NULLS LAST;
