SELECT
  me.chemical_system,
  COUNT(DISTINCT me.entry_id) AS compound_count,
  ARRAY_AGG(DISTINCT me.reduced_formula ORDER BY me.reduced_formula) AS compounds
FROM material_entry AS me
WHERE me.number_of_elements = 2
  AND (
    EXISTS (
      SELECT 1
      FROM structure AS s
      LEFT JOIN prototype_definition AS pd
        ON s.prototype = pd.prototype_id
      WHERE s.entry_id = me.entry_id
        AND (
          REGEXP_REPLACE(LOWER(REPLACE(COALESCE(s.strukturbericht, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
          OR REGEXP_REPLACE(LOWER(REPLACE(COALESCE(pd.strukturbericht, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
          OR REGEXP_REPLACE(LOWER(REPLACE(COALESCE(s.prototype, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
          OR REGEXP_REPLACE(LOWER(REPLACE(COALESCE(pd.prototype_name, ''), '₂', '2')), '[^a-z0-9]', '', 'g') LIKE '%l12%'
        )
    )
    OR EXISTS (
      SELECT 1
      FROM formation_enthalpy AS fh
      WHERE fh.entry_id = me.entry_id
        AND (
          REGEXP_REPLACE(LOWER(REPLACE(COALESCE(fh.strukturbericht, ''), '₂', '2')), '[^a-z0-9]', '', 'g') = 'l12'
          OR REGEXP_REPLACE(LOWER(REPLACE(COALESCE(fh.prototype, ''), '₂', '2')), '[^a-z0-9]', '', 'g') LIKE '%l12%'
        )
    )
  )
GROUP BY me.chemical_system
ORDER BY me.chemical_system;
