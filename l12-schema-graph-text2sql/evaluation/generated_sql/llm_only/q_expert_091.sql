SELECT
  CASE
    WHEN UPPER(REPLACE(strukturbericht, '_', '')) = 'L12' THEN 'L12'
    WHEN UPPER(strukturbericht) = 'B2' THEN 'B2'
  END AS strukturbericht,
  COUNT(*) AS total_compounds,
  COUNT(*) FILTER (WHERE is_stable = TRUE) AS stable_compounds,
  COUNT(*) FILTER (WHERE is_stable = TRUE)::double precision / NULLIF(COUNT(*), 0) AS stable_fraction
FROM formation_enthalpy
WHERE UPPER(REPLACE(strukturbericht, '_', '')) = 'L12'
   OR UPPER(strukturbericht) = 'B2'
GROUP BY 1
ORDER BY 1;
