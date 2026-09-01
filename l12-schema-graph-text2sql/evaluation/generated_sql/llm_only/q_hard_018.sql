SELECT
  COUNT(DISTINCT fe.entry_id) FILTER (WHERE fe.is_stable = TRUE) AS stable_count,
  COUNT(DISTINCT fe.entry_id) FILTER (WHERE fe.is_stable = FALSE) AS not_stable_count
FROM formation_enthalpy fe
WHERE fe.strukturbericht = 'L1_2'
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = fe.entry_id
      AND c.element = 'Al'
  );
