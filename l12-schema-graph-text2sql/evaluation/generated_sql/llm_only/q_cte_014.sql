WITH l12_compounds AS (
  SELECT DISTINCT
    entry_id,
    (enthalpy_vs_element_ground_states < 0) AS is_negative
  FROM formation_enthalpy
  WHERE enthalpy_vs_element_ground_states IS NOT NULL
    AND (
      strukturbericht = 'L1_2'
      OR strukturbericht = 'L12'
      OR prototype ILIKE '%L1_2%'
      OR prototype ILIKE '%L12%'
    )
),
compound_elements AS (
  SELECT DISTINCT
    c.element,
    c.entry_id
  FROM composition c
  JOIN l12_compounds l
    ON l.entry_id = c.entry_id
)
SELECT
  ce.element,
  AVG(CASE WHEN l.is_negative THEN 1.0 ELSE 0.0 END) AS negative_fraction
FROM compound_elements ce
JOIN l12_compounds l
  ON l.entry_id = ce.entry_id
GROUP BY ce.element
ORDER BY ce.element;
