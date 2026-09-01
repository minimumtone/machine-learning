WITH l12_entries AS (
  SELECT s.entry_id
  FROM structure s
  LEFT JOIN prototype_definition pd
    ON s.prototype = pd.prototype_id
  WHERE regexp_replace(upper(coalesce(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
     OR regexp_replace(upper(coalesce(s.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
     OR regexp_replace(upper(coalesce(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
     OR regexp_replace(upper(coalesce(pd.prototype_name, '')), '[^A-Z0-9]', '', 'g') = 'L12'

  UNION

  SELECT fe.entry_id
  FROM formation_enthalpy fe
  WHERE regexp_replace(upper(coalesce(fe.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
     OR regexp_replace(upper(coalesce(fe.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
)
SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  COUNT(DISTINCT mr.reference_id) AS literature_reference_count
FROM material_entry me
JOIN l12_entries le
  ON me.entry_id = le.entry_id
JOIN material_reference mr
  ON me.entry_id = mr.entry_id
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula
HAVING COUNT(DISTINCT mr.reference_id) >= 3
ORDER BY
  literature_reference_count DESC,
  me.reduced_formula,
  me.entry_id;
