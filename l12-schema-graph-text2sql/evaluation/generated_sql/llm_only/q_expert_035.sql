SELECT
  COUNT(*) FILTER (WHERE ps.is_stable = FALSE)::double precision / NULLIF(COUNT(*), 0) AS non_stable_fraction
FROM phase_stability ps
JOIN structure s
  ON s.entry_id = ps.entry_id
LEFT JOIN prototype_definition pd
  ON pd.prototype_id = s.prototype
WHERE
  s.strukturbericht = 'B1'
  OR pd.strukturbericht = 'B1'
  OR s.prototype ILIKE '%NaCl%'
  OR pd.prototype_name ILIKE '%NaCl%';
