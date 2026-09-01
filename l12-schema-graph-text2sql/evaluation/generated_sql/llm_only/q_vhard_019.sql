SELECT
  fh.entry_id,
  fh.formula,
  fh.reduced_formula,
  fh.reference_set,
  fh.formation_enthalpy_ev_per_atom + fh.weighted_element_delta_e AS rereferenced_formation_energy_ev_per_atom
FROM formation_enthalpy AS fh
WHERE fh.is_stable = TRUE
  AND EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = fh.entry_id
      AND c.element = 'Ni'
  )
  AND (
    regexp_replace(replace(upper(coalesce(fh.strukturbericht, '')), '₂', '2'), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(replace(upper(coalesce(fh.prototype, '')), '₂', '2'), '[^A-Z0-9]', '', 'g') = 'L12'
    OR EXISTS (
      SELECT 1
      FROM structure AS s
      WHERE s.entry_id = fh.entry_id
        AND (
          regexp_replace(replace(upper(coalesce(s.strukturbericht, '')), '₂', '2'), '[^A-Z0-9]', '', 'g') = 'L12'
          OR regexp_replace(replace(upper(coalesce(s.prototype, '')), '₂', '2'), '[^A-Z0-9]', '', 'g') = 'L12'
        )
    )
  )
  AND fh.formation_enthalpy_ev_per_atom IS NOT NULL
  AND fh.weighted_element_delta_e IS NOT NULL
ORDER BY rereferenced_formation_energy_ev_per_atom ASC
LIMIT 5;
