WITH candidates AS (
  SELECT
    c.element AS a_site_element,
    fh.entry_id,
    fh.formula,
    fh.reduced_formula,
    fh.enthalpy_vs_element_ground_states AS formation_energy_ev_per_atom,
    ROW_NUMBER() OVER (
      PARTITION BY c.element
      ORDER BY fh.enthalpy_vs_element_ground_states ASC, fh.entry_id
    ) AS rn
  FROM formation_enthalpy fh
  JOIN composition c
    ON c.entry_id = fh.entry_id
  LEFT JOIN structure s
    ON s.entry_id = fh.entry_id
  WHERE c.site_label = 'A-site'
    AND fh.enthalpy_vs_element_ground_states IS NOT NULL
    AND (
      fh.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR fh.prototype ILIKE '%L1_2%'
      OR fh.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L12%'
    )
)
SELECT
  a_site_element,
  entry_id,
  formula,
  reduced_formula,
  formation_energy_ev_per_atom
FROM candidates
WHERE rn = 1
ORDER BY a_site_element;
